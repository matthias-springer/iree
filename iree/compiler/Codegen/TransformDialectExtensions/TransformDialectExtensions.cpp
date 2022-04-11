// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TransformDialectExtensions.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/TransformOpInterface.h"
#include "iree-dialects/Transforms/Functional.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE;

//===---------------------------------------------------------------------===//
// Patterns for InParallelToHAL rewrite.
//===---------------------------------------------------------------------===//

/// Pattern to rewrite a InParallelOp to the HAL dialect.
/// This is written in a way that allows rewriting a single InParallelOp at a
/// time. Atm this is used in a global rewrite of all InParallelOps in a region
/// to properly account for ordering requirements (i.e. innermost InParallelOp
/// need to be rewritten before outermost ones).
/// In the future, finer-grained single op control may be better.
struct InParallelOpToHALRewriter
    : public OpRewritePattern<LinalgExt::InParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<SmallVector<Operation *>> returningMatchAndRewrite(
      LinalgExt::InParallelOp inParallelOp, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(LinalgExt::InParallelOp inParallelOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(inParallelOp, rewriter);
  }
};

static int64_t getNumEnclosingInParallelOps(Operation *op) {
  int64_t numInParallelOps = 0;
  while (auto parentOp = op->getParentOfType<LinalgExt::InParallelOp>()) {
    op = parentOp;
    ++numInParallelOps;
  }
  return numInParallelOps;
}

/// Return the unique HALExecutableEntryPointOp within parentFuncOp or creates
/// a new op whose terminator returns the triple (one, one, one).
/// This is a placeholder into which more information can be inserted to build
/// the proper workgroup counts.
/// Return nullptr if the parentFuncOp contains more than a single
/// HALExecutableEntryPointOp.
// TODO: This will not be neded once transform dialect can use real HAL ops.
static HAL::ExecutableEntryPointOp ensureEntryPointOpWithBody(
    PatternRewriter &rewriter, HAL::ExecutableVariantOp variantOp) {
  HAL::ExecutableEntryPointOp entryPointOp;
  WalkResult res = variantOp.walk([&](HAL::ExecutableEntryPointOp op) {
    if (entryPointOp) return WalkResult::interrupt();
    entryPointOp = op;
    return WalkResult::advance();
  });
  assert(entryPointOp && !res.wasInterrupted() &&
         "expected one single entry point");
  if (res.wasInterrupted()) return nullptr;

  if (entryPointOp.getWorkgroupCountBody()) return entryPointOp;

  Location loc = entryPointOp.getLoc();
  int64_t numWorkgroupDims = HAL::ExecutableEntryPointOp::getNumWorkgroupDims();
  Block &block = entryPointOp.workgroup_count_region().emplaceBlock();
  for (int64_t i = 0; i < numWorkgroupDims; ++i)
    block.addArgument(rewriter.getIndexType(), loc);

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(&block);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> workgroupCounts(numWorkgroupDims, one);
  rewriter.create<HAL::ReturnOp>(loc, TypeRange{}, workgroupCounts);
  return entryPointOp;
}

FailureOr<SmallVector<Operation *>>
InParallelOpToHALRewriter::returningMatchAndRewrite(
    LinalgExt::InParallelOp inParallelOp, PatternRewriter &rewriter) const {
  if (!inParallelOp->getParentOfType<HAL::ExecutableVariantOp>())
    return inParallelOp->emitError("No enclosing HAL::ExecutableVariantOp");

  // Only allow rewriting to inParallelOp late, i.e. after bufferization. This
  // avoids layering issues and fishy forwardings of tensor.insert/extract to
  // flow.tensor.load/store.
  // These forwardings are fishy because they are considered "canonicalizations"
  // which should have no effects on semantics; yet, if we omit them, we end up
  // with racy tensor semantics.
  // The InParallel abstraction correctly models tensors + tiling + parallelism.
  if (inParallelOp.getNumResults() > 0) {
    return inParallelOp->emitError(
        "Only allow rewriting 0-result inParallelOp (i.e. call iree_bufferize "
        "before lowering)");
  }

  // Rewriter-based RAUW operates on Operation* atm, bail if we can't get it.
  Operation *numThreadDefiningOp = inParallelOp.num_threads().getDefiningOp();
  if (!numThreadDefiningOp) {
    return inParallelOp->emitError(
        "Cannot find a defining op for the num_threads operand");
  }

  // Rewrites must happen bottom-up to get the proper workgroup id ordering.
  LinalgExt::InParallelOp nestedInParallelOp;
  WalkResult walkResult = inParallelOp.walk([&](LinalgExt::InParallelOp op) {
    nestedInParallelOp = op;
    return (op == inParallelOp) ? WalkResult::advance()
                                : WalkResult::interrupt();
  });
  if (walkResult.wasInterrupted()) {
    return inParallelOp->emitError(
               "Failed to rewrite top-level InParallelOp with nested "
               "InParallelOp:\n")
           << *nestedInParallelOp.getOperation();
  }

  // #of enclosing InParallelOp determine the #idx in:
  //   hal.interface.workgroup.id[#idx] : index
  //   hal.interface.workgroup.count[#idx] : index
  int64_t numEnclosingInParallelOps =
      getNumEnclosingInParallelOps(inParallelOp);
  if (numEnclosingInParallelOps >=
      HAL::ExecutableEntryPointOp::getNumWorkgroupDims()) {
    return inParallelOp->emitError(
        "Too many InParallelOps, exceeds the maximum number of workgroup dims");
  }

  // Custom hal.executable.entry_point.
  // TODO: pull in the proper operands as the bbArgs to allow dynamic sizes.
  HAL::ExecutableVariantOp variantOp =
      inParallelOp->getParentOfType<HAL::ExecutableVariantOp>();
  auto region = std::make_unique<Region>();
  HAL::ExecutableEntryPointOp entryPointOp =
      ensureEntryPointOpWithBody(rewriter, variantOp);

  // At this point, the region is known to have a body.
  HAL::ReturnOp returnOp = cast<HAL::ReturnOp>(
      entryPointOp.getWorkgroupCountBody()->getTerminator());
  // Update the numEnclosingInParallelOps^th operand with an in-body clone of
  // numThreadDefiningOp.
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(returnOp);
    // TODO: This can only work on constant ops atm. In the future, handle
    // copying full backward slices (but we'll need a better
    // HALExecutableEntryPointOp bbargs contract).
    Operation *op = rewriter.clone(*numThreadDefiningOp);
    rewriter.startRootUpdate(returnOp);
    SmallVector<Value> operands = returnOp->getOperands();
    // TODO: ensure this is already 1 or the same value otherwise we are in
    // presence of sibling InParallelOp's that are incompatible.
    operands[numEnclosingInParallelOps] = op->getResult(0);
    returnOp->setOperands(operands);
    rewriter.finalizeRootUpdate(returnOp);
  }

  Location loc = inParallelOp.getLoc();
  auto idOp = rewriter.create<HAL::InterfaceWorkgroupIDOp>(
      loc, numEnclosingInParallelOps);
  auto countOp = rewriter.create<HAL::InterfaceWorkgroupCountOp>(
      loc, numEnclosingInParallelOps);

  // Get a reference to the terminator that will subsequently be moved.
  LinalgExt::PerformConcurrentlyOp performConcurrentlyOp =
      inParallelOp.getTerminator();

  // First, update the uses of num_threads() within the inParallelOp block.
  rewriter.replaceOpWithinBlock(numThreadDefiningOp, countOp.result(),
                                &inParallelOp.region().front());

  // Steal the iree_compiler::LinalgExt::InParallel ops, right before
  // the inParallelOp. Replace the bbArg by the HAL id.
  // This includes the terminator `performConcurrentlyOp` that still needs to
  // be erased separately.
  SmallVector<Value> bbArgsTranslated{idOp.result()};
  rewriter.mergeBlockBefore(&inParallelOp.region().front(), inParallelOp,
                            bbArgsTranslated);

  assert(inParallelOp->getNumResults() == 0 && "Expects 0-result inParallelOp");
  rewriter.eraseOp(performConcurrentlyOp);
  rewriter.eraseOp(inParallelOp);
  return SmallVector<Operation *>();
}

//===---------------------------------------------------------------------===//
// Default allocation functions for CPU backend
// TODO: register the bufferization behavior in a target-specific way.
//===---------------------------------------------------------------------===//

// Default allocation function to use with IREEs bufferization.
static Value cpuAllocationFunction(OpBuilder &builder, Location loc,
                                   ArrayRef<int64_t> staticShape,
                                   Type elementType,
                                   ArrayRef<Value> dynamicSizes) {
  MemRefType allocType = MemRefType::get(staticShape, elementType);
  return builder.create<memref::AllocaOp>(loc, allocType, dynamicSizes);
}

// Allocation callbacks to use with upstream comprehensive bufferization
static FailureOr<Value> cpuComprehensiveBufferizeAllocationFn(
    OpBuilder &builder, Location loc, MemRefType memRefType,
    ValueRange dynamicSizes, unsigned alignment) {
  return builder
      .create<memref::AllocaOp>(loc, memRefType, dynamicSizes,
                                builder.getI64IntegerAttr(alignment))
      .getResult();
}

static LogicalResult cpuComprehensiveBufferizeDeallocationFn(OpBuilder &builder,
                                                             Location loc,
                                                             Value allocation) {
  return success();
}

static LogicalResult cpuComprehensiveBufferizeCopyFn(OpBuilder &builder,
                                                     Location loc, Value from,
                                                     Value to) {
  // TODO: ideally we should use linalg.copy which was recently reintroduced
  // as an OpDSL named op. However, IREE-specific patterns to cleanup spurious
  // post-bufferization copies do not trigger properly.
  // So we keep using `createLinalgCopyOp` which builds a GenericOp.
  builder.create<linalg::CopyOp>(loc, from, to);
  // mlir::iree_compiler::createLinalgCopyOp(builder, loc, from, to);
  return success();
}

//===---------------------------------------------------------------------===//
// IREE-specific transformations defined outside of iree_linalg_transform.
//===---------------------------------------------------------------------===//

// Note: with the recent TypeID changes, hiding these classes inside an
// anonymous namespace would require specific `MLIR_DECLARE_EXPLICIT_TYPE_ID`
// for each class.

// namespace {

// TODO: Move to tablegen. Until this stabilizes upstream, simple C++ is
// enough.
class IREEBufferizeOp
    : public Op<IREEBufferizeOp,
                linalg::transform::TransformOpInterface::Trait> {
 public:
  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static constexpr llvm::StringLiteral getOperationName() {
    return llvm::StringLiteral("iree_linalg_transform.iree_bufferize");
  }

  Value target() { return nullptr; }

  LogicalResult apply(linalg::transform::TransformResults &results,
                      linalg::transform::TransformState &state) {
    PassManager pm(getContext());
    // Bufferize the dispatch.
    using mlir::bufferization::BufferizationOptions;
    BufferizationOptions::AllocationFn allocationFn =
        cpuComprehensiveBufferizeAllocationFn;
    BufferizationOptions::DeallocationFn deallocationFn =
        cpuComprehensiveBufferizeDeallocationFn;
    BufferizationOptions::MemCpyFn memcpyFn = cpuComprehensiveBufferizeCopyFn;
    mlir::iree_compiler::addIREEComprehensiveBufferizePasses(
        pm, allocationFn, deallocationFn, memcpyFn);
    WalkResult res = state.getTopLevel()->walk([&](ModuleOp moduleOp) {
      if (failed(pm.run(moduleOp))) return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return failure(res.wasInterrupted());
  }

  // let assemblyFormat = "attr-dict";
  static ParseResult parse(OpAsmParser &parser, OperationState &state) {
    parser.parseOptionalAttrDict(state.attributes);
    return success();
  }

  // let assemblyFormat = "attr-dict";
  void print(OpAsmPrinter &printer) {
    printer.printOptionalAttrDict((*this)->getAttrs());
  }
};

// TODO: Move to tablegen. Until this stabilizes upstream, simple C++ is
// enough.
class IREESetNumWorkgroupToOneOp
    : public Op<IREESetNumWorkgroupToOneOp,
                linalg::transform::TransformOpInterface::Trait> {
 public:
  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static constexpr llvm::StringLiteral getOperationName() {
    return llvm::StringLiteral(
        "iree_linalg_transform.iree_set_num_workgroups_to_one");
  }

  Value target() { return nullptr; }

  LogicalResult apply(linalg::transform::TransformResults &results,
                      linalg::transform::TransformState &state) {
    auto variantOp = dyn_cast<HAL::ExecutableVariantOp>(state.getTopLevel());
    if (!variantOp) return failure();
    return iree_compiler::setNumWorkgroupsImpl(variantOp, {});
  }

  // let assemblyFormat = "attr-dict";
  static ParseResult parse(OpAsmParser &parser, OperationState &state) {
    parser.parseOptionalAttrDict(state.attributes);
    return success();
  }

  // let assemblyFormat = "attr-dict";
  void print(OpAsmPrinter &printer) {
    printer.printOptionalAttrDict((*this)->getAttrs());
  }
};

// TODO: Move to tablegen. Until this stabilizes upstream, simple C++ is
// enough.
class IREELinalgExtInParallelToHALOp
    : public Op<IREELinalgExtInParallelToHALOp,
                linalg::transform::TransformOpInterface::Trait> {
 public:
  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static constexpr llvm::StringLiteral getOperationName() {
    return llvm::StringLiteral(
        "iree_linalg_transform.iree_linalg_ext_inparallel_to_hal");
  }

  Value target() { return nullptr; }

  LogicalResult apply(linalg::transform::TransformResults &results,
                      linalg::transform::TransformState &state) {
    if (state.getTopLevel()
            ->walk<WalkOrder::PostOrder>([&](LinalgExt::InParallelOp op) {
              if (failed(functional::applyReturningPatternAt(
                      InParallelOpToHALRewriter(getContext()), op)))
                return WalkResult::interrupt();
              return WalkResult::advance();
            })
            .wasInterrupted())
      return failure();

    // Apply post-distribution canonicalization passes.
    // TODO: these should be done like other transform dialect
    // canonicalizations that preserve IR handles.
    RewritePatternSet canonicalization(getContext());
    AffineApplyOp::getCanonicalizationPatterns(canonicalization, getContext());
    AffineMinOp::getCanonicalizationPatterns(canonicalization, getContext());
    iree_compiler::populateAffineMinSCFCanonicalizationPattern(
        canonicalization);
    // TODO: Careful. forwarding to Flow is more than a simple canonicalization.
    Flow::populateFlowDispatchCanonicalizationPatterns(canonicalization,
                                                       getContext());
    return applyPatternsAndFoldGreedily(state.getTopLevel(),
                                        std::move(canonicalization));
  }

  // let assemblyFormat = "attr-dict";
  static ParseResult parse(OpAsmParser &parser, OperationState &state) {
    parser.parseOptionalAttrDict(state.attributes);
    return success();
  }

  // let assemblyFormat = "attr-dict";
  void print(OpAsmPrinter &printer) {
    printer.printOptionalAttrDict((*this)->getAttrs());
  }
};

/// Test extension of the Transform dialect. Registers additional ops and
/// declares PDL as dependent dialect since the additional ops are using PDL
/// types for operands and results.
class LinalgTransformDialectExtension
    : public mlir::linalg::transform::TransformDialectExtension<
          LinalgTransformDialectExtension> {
 public:
  LinalgTransformDialectExtension() {
    declareDependentDialect<pdl::PDLDialect>();
    // clang-format off
    registerTransformOps<IREEBufferizeOp, 
                         IREESetNumWorkgroupToOneOp,
                         IREELinalgExtInParallelToHALOp>();
    // clang-format on
    // TODO: hook up to Tablegen.
    //     registerTransformOps<
    // #define GET_OP_LIST
    // #include "LinalgTransformDialectExtension.cpp.inc"
    //         >();
  }
};

// } // namespace anonymous

void mlir::iree_compiler::registerLinalgTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<LinalgTransformDialectExtension>();
}
