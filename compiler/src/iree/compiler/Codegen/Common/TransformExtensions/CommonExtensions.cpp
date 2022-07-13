// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CommonExtensions.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/ListenerGreedyPatternRewriteDriver.h"
#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"
#include "iree/compiler/Codegen/Passes.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;

iree_compiler::IREE::transform_dialect::CommonExtensions::CommonExtensions() {
  registerTransformOps<
#define GET_OP_LIST
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensionsOps.cpp.inc"
      >();
}

void mlir::iree_compiler::registerTransformDialectCommonExtension(
    DialectRegistry &registry) {
  registry.addExtensions<transform_dialect::CommonExtensions>();
}

//===---------------------------------------------------------------------===//
// MatchConstraintsOpInterface
//===---------------------------------------------------------------------===//

LogicalResult transform_dialect::detail::verifyMatchConstraintOpInterface(
    Operation *op) {
  if (op->getNumResults() > 0)
    return op->emitOpError("constraints cannot have results");
  return success();
}

//===---------------------------------------------------------------------===//
// ApplyPatternsOp
//===---------------------------------------------------------------------===//

static void addRankReducingPatterns(RewritePatternSet &patterns) {
  vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
  linalg::populateFoldUnitExtentDimsPatterns(patterns);
}

static void addAllRegisteredCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  for (Dialect *dialect : ctx->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (RegisteredOperationName op : ctx->getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, ctx);
}

DiagnosedSilenceableFailure transform_dialect::ApplyPatternsOp::applyToOne(
    Operation *target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    target->emitOpError(
        "applies only to isolated-from-above targets because it needs to apply "
        "patterns greedily");
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  }
  MLIRContext *ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  if (getCanonicalization()) addAllRegisteredCanonicalizationPatterns(patterns);
  if (getRankReducing()) addRankReducingPatterns(patterns);

  TrackingListener listener(state);
  GreedyRewriteConfig config;
  LogicalResult result = applyPatternsAndFoldGreedily(
      target, std::move(patterns), config, &listener);
  LogicalResult listenerResult = listener.checkErrorState();
  if (failed(result) || failed(listenerResult))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  results.assign({target});
  return DiagnosedSilenceableFailure(success());
}

//===---------------------------------------------------------------------===//
// IREEBufferizeOp
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// Default allocation functions for CPU backend
// TODO: register the bufferization behavior in a target-specific way.
// TODO: Maybe bufferize should have a separate cpu and a gpu version. This is
// unclear though: what happens on heterogeneous HW ?
//===---------------------------------------------------------------------===//

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
  // builder.create<linalg::CopyOp>(loc, from, to);
  mlir::iree_compiler::createLinalgCopyOp(builder, loc, from, to);
  return success();
}

static FailureOr<Value> gpuComprehensiveBufferizeAllocationFn(
    OpBuilder &builder, Location loc, MemRefType memRefType,
    ValueRange dynamicSizes, unsigned alignment) {
  // TODO: use gpu::GPUDialect::getWorkgroupAddressSpace() but this requires
  // moving out of CommonExtensions.
  MemRefType allocType = MemRefType::get(memRefType.getShape(),
                                         memRefType.getElementType(), {}, 3);
  return builder
      .create<memref::AllocOp>(loc, allocType, dynamicSizes,
                               builder.getI64IntegerAttr(alignment))
      .getResult();
}

static LogicalResult gpuComprehensiveBufferizeDeallocationFn(OpBuilder &builder,
                                                             Location loc,
                                                             Value allocation) {
  builder.create<memref::DeallocOp>(loc, allocation);
  return success();
}

static LogicalResult gpuComprehensiveBufferizeCopyFn(OpBuilder &builder,
                                                     Location loc, Value from,
                                                     Value to) {
  // TODO: ideally we should use linalg.copy which was recently reintroduced
  // as an OpDSL named op. However, IREE-specific patterns to cleanup spurious
  // post-bufferization copies do not trigger properly.
  // So we keep using `createLinalgCopyOp` which builds a GenericOp.
  // builder.create<linalg::CopyOp>(loc, from, to);
  mlir::iree_compiler::createLinalgCopyOp(builder, loc, from, to);
  return success();
}

DiagnosedSilenceableFailure transform_dialect::IREEBufferizeOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  PassManager pm(getContext());
  // Bufferize the dispatch.
  using mlir::bufferization::BufferizationOptions;
  BufferizationOptions::AllocationFn allocationFn =
      cpuComprehensiveBufferizeAllocationFn;
  BufferizationOptions::DeallocationFn deallocationFn =
      cpuComprehensiveBufferizeDeallocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = cpuComprehensiveBufferizeCopyFn;
  if (getTargetGpu()) {
    allocationFn = gpuComprehensiveBufferizeAllocationFn;
    deallocationFn = gpuComprehensiveBufferizeDeallocationFn;
    memcpyFn = gpuComprehensiveBufferizeCopyFn;
  }
  mlir::iree_compiler::addIREEComprehensiveBufferizePasses(
      pm, allocationFn, deallocationFn, memcpyFn);
  WalkResult res = state.getTopLevel()->walk([&](ModuleOp moduleOp) {
    if (failed(pm.run(moduleOp))) {
      getOperation()->emitError()
          << "failed to bufferize ModuleOp:\n"
          << *(moduleOp.getOperation()) << "\nunder top-level:\n"
          << *state.getTopLevel();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return DiagnosedSilenceableFailure(failure(res.wasInterrupted()));
}

//===---------------------------------------------------------------------===//
// MatchOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform_dialect::MatchOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  Block *body = getBody();
  ArrayRef<Operation *> payloadOps = state.getPayloadOps(getTarget());
  if (payloadOps.size() != 1)
    return DiagnosedSilenceableFailure(
        this->emitOpError("requires exactly one target handle"));

  SmallVector<Operation *> res;
  payloadOps.front()->walk([&](Operation *candidate) {
    // Look for nested ops.
    if (candidate == payloadOps.front())
      return WalkResult::advance();

    // If no constraints are specified, all ops are returned.
    if (!body) {
      res.push_back(candidate);
      return WalkResult::advance();
    }

    // Include an op if is satisfies all constraints.
    if (all_of(body->without_terminator(), [&](Operation &matchConstraintOp) {
          return cast<MatchConstraintOpInterface>(&matchConstraintOp)
              .satisfied(candidate);
        }))
      res.push_back(candidate);

    return WalkResult::advance();
  });

  results.set(getResult().cast<OpResult>(), res);
  return DiagnosedSilenceableFailure(success());
}

ParseResult transform_dialect::MatchOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  Type pdlOpType = parser.getBuilder().getType<pdl::OperationType>();
  OpAsmParser::UnresolvedOperand matchedOpBbArg, targetOperand;
  bool hasRegion = failed(parser.parseOptionalKeyword("within"));

  if (hasRegion) {
    if (parser.parseOperand(matchedOpBbArg) || parser.parseKeyword("within") ||
        parser.parseOperand(targetOperand))
      return failure();
  } else {
    if (parser.parseOperand(targetOperand)) return failure();
  }

  Region *body = result.addRegion();
  if (hasRegion) {
    if (parser.parseKeyword("where")) return failure();
    SmallVector<OpAsmParser::Argument, 4> regionArgs;
    auto &arg = regionArgs.emplace_back();
    arg.ssaName = matchedOpBbArg;
    arg.type = pdlOpType;
    if (parser.parseRegion(*body, regionArgs)) return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  if (parser.resolveOperand(targetOperand, pdlOpType, result.operands))
    return failure();

  result.addTypes(pdlOpType);
  return success();
}

void transform_dialect::MatchOp::print(OpAsmPrinter &p) {
  if (getBody()) {
    p << " " << getMatchedOp() << " within " << getTarget() << " ";
    p.printRegion(getConstraints(), /*printEntryBlockArgs=*/false);
  } else {
    p << " within " << getTarget();
  }
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

LogicalResult transform_dialect::MatchOp::verify() {
  if (getConstraints().getBlocks().size() > 1)
    return this->emitOpError("must not have more than 1 block");
  if (getBody()) {
    if (!isa<transform_dialect::MatchTerminatorOp>(getBody()->getTerminator()))
      return this->emitError("requires match.terminator terminator");
    for (Operation &op : getBody()->without_terminator())
      if (!isa<MatchConstraintOpInterface>(&op))
        return op.emitError("body ops must be constraints");
  }
  return success();
}

//===---------------------------------------------------------------------===//
// MatchIsAOp
//===---------------------------------------------------------------------===//

LogicalResult transform_dialect::MatchIsAOp::verify() {
  bool opXorIface = getMatchOp().hasValue() ^ getMatchInterface().hasValue();
  if (!opXorIface)
    return this->emitOpError(
        "requires a either a match_op or a match_interface attribute (but not "
        "both)");
  return success();
}

bool transform_dialect::MatchIsAOp::satisfied(Operation *op) {
  llvm::StringSet<> strs;
  if (getMatchOp().hasValue())
    strs.insert(getMatchOp()->getAsValueRange<StringAttr>().begin(),
                getMatchOp()->getAsValueRange<StringAttr>().end());

  if (strs.contains(op->getName().getStringRef())) return true;

  // Interfaces cannot be matched by name, just by ID.
  // So we specifically encode the interfaces we care about for this op.
  if (getMatchInterface().hasValue()) {
    auto iface = getMatchInterface().getValue();
    if (iface == transform_dialect::MatchInterfaceEnum::LinalgOp &&
        isa<linalg::LinalgOp>(op))
      return true;
    if (iface == transform_dialect::MatchInterfaceEnum::TilingInterface &&
        isa<TilingInterface>(op))
      return true;
  }

  return false;
}

#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensionsAttrs.cpp.inc"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensionsInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensionsOps.cpp.inc"
