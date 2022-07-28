// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "FlowExtensions.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgTransform/SimplePatternRewriter.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/DispatchLinalgOnTensors.h"
#include "iree/compiler/Dialect/Flow/Transforms/FusionUtils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;

iree_compiler::IREE::transform_dialect::FlowExtensions::FlowExtensions() {
  registerTransformOps<
#define GET_OP_LIST
#include "iree/compiler/Dialect/Flow/TransformExtensions/FlowExtensionsOps.cpp.inc"
      >();
}

void mlir::iree_compiler::registerTransformDialectFlowExtension(
    DialectRegistry &registry) {
  registry.addExtensions<transform_dialect::FlowExtensions>();
}

// TODO: Upstream to ShapeType and reuse.
static SmallVector<int64_t> getIndicesOfDynamicDims(ShapedType t) {
  int64_t numDynamicDims = t.getNumDynamicDims();
  SmallVector<int64_t> res(numDynamicDims);
  for (int64_t dim = 0; dim != numDynamicDims; ++dim)
    res[dim] = t.getDynamicDimIndex(dim);
  return res;
}

//===---------------------------------------------------------------------===//
// ForeachThreadToFlowDispatchWorkgroupsOp
//===---------------------------------------------------------------------===//

/// Populate the workgroup_count region of `dispatchOp`.
/// For now, this only supports constant index ops and empty workload operands.
/// Assumes the Flow::DispatchWorkgroupsOp is built with an empty region.
static LogicalResult populateWorkgroupCountComputingRegion(
    PatternRewriter &rewriter, scf::ForeachThreadOp foreachThreadOp,
    Flow::DispatchWorkgroupsOp dispatchOp) {
  Location loc = foreachThreadOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  Region &r = dispatchOp.getWorkgroupCount();
  assert(r.empty() && "expected block-less workgroup_count region");
  Block *block = rewriter.createBlock(&r);
  rewriter.setInsertionPointToStart(block);

  SmallVector<Value> results;
  // For now, this assumes that we only pull in constants.
  // TODO: Iteratively pull operations that are only consuming IndexType.
  for (Value v : foreachThreadOp.getNumThreads()) {
    auto op = dyn_cast_or_null<arith::ConstantIndexOp>(v.getDefiningOp());
    if (!op) return failure();
    results.push_back(
        cast<arith::ConstantIndexOp>(rewriter.clone(*op)).getResult());
  }
  // Resize to `3` to match IREE's assumptions.
  for (unsigned i = results.size(); i < 3; ++i) {
    results.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
  }
  rewriter.create<Flow::ReturnOp>(loc, results);

  return success();
}

/// Rewrite ParallelInsertSlice ops in `performConcurrentlyOp` as Flow
/// DispatchTensorStoreOps.
/// Ops are inserted just before the `block` terminator.
static void rewriteParallelInsertSlices(
    PatternRewriter &rewriter, scf::PerformConcurrentlyOp performConcurrentlyOp,
    Block &block, ValueRange resultTensorOperands,
    ValueRange resultTensorsDynamicDims, BlockAndValueMapping tensorToFlowBvm) {
  Location loc = performConcurrentlyOp.getLoc();
  int64_t resultIndex = 0;
  for (const Operation &yieldingOp :
       llvm::make_early_inc_range(performConcurrentlyOp.getYieldingOps())) {
    auto parallelInsertOp = cast<tensor::ParallelInsertSliceOp>(&yieldingOp);
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(block.getTerminator());
    auto dynamicDims = Util::findVariadicDynamicDims(
        resultIndex, resultTensorOperands, resultTensorsDynamicDims);
    // clang-format off
    rewriter.create<Flow::DispatchTensorStoreOp>(
        loc,
        parallelInsertOp.getSource(),
        tensorToFlowBvm.lookup(parallelInsertOp.getDest()),
        dynamicDims,
        parallelInsertOp.getMixedOffsets(),
        parallelInsertOp.getMixedSizes(),
        parallelInsertOp.getMixedStrides());
    // clang-format on
    ++resultIndex;
    rewriter.eraseOp(parallelInsertOp);
  }
}

/// Rewrite ExtractSlice ops in `dispatchOp` as Flow::DispatchTensorLoadOps.
/// Takes a list of all tensor and all tensorDynamicDims operands to the
/// dispatchOp as well as a BlockAndValueMapping from tensor operands to the
/// corresponding Flow dispatch tensor bbArgs.
static void rewriteExtractSlices(PatternRewriter &rewriter,
                                 Flow::DispatchWorkgroupsOp dispatchOp,
                                 ValueRange tensorOperands,
                                 ValueRange tensorDynamicDims,
                                 BlockAndValueMapping tensorToFlowBvm) {
  dispatchOp->walk([&](tensor::ExtractSliceOp extractSliceOp) {
    Value source = extractSliceOp.getSource();
    auto it = llvm::find(tensorOperands, source);
    if (it == tensorOperands.end()) return;
    int64_t index = std::distance(tensorOperands.begin(), it);
    Value sourceFlow = tensorToFlowBvm.lookupOrNull(source);
    if (!sourceFlow) return;

    Location loc = extractSliceOp.getLoc();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(extractSliceOp);
    auto dynamicDims =
        Util::findVariadicDynamicDims(index, tensorOperands, tensorDynamicDims);
    // clang-format off
    Value load = rewriter.create<Flow::DispatchTensorLoadOp>(
        loc,
        sourceFlow,
        dynamicDims,
        extractSliceOp.getMixedOffsets(),
        extractSliceOp.getMixedSizes(),
        extractSliceOp.getMixedStrides());
    // clang-format on
    rewriter.replaceOp(extractSliceOp, load);
  });
}

static void cloneOpsIntoForeachThreadOp(RewriterBase &rewriter,
                                        scf::ForeachThreadOp foreachThreadOp) {
  // 1. Find all ops that should be cloned into the ForeachThreadOp.
  llvm::SetVector<Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(foreachThreadOp.getRegion(),
                                  valuesDefinedAbove);
  // Add all ops who's results are used inside the ForeachThreadOp to the
  // worklist.
  llvm::SetVector<Operation *> worklist;
  for (Value v : valuesDefinedAbove)
    if (Operation *op = v.getDefiningOp()) worklist.insert(op);
  llvm::SmallVector<Operation *> opsToClone;
  llvm::DenseSet<Operation *> visited;

  // Process all ops in the worklist.
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (visited.contains(op)) continue;
    visited.insert(op);

    // Do not clone ops that are not clonable.
    if (!mlir::iree_compiler::IREE::Flow::isClonableIntoDispatchOp(op))
      continue;

    // Do not clone ParallelInsertSliceOp destinations.
    bool isDestination =
        any_of(foreachThreadOp.getTerminator().getYieldingOps(),
               [&](Operation &insertOp) {
                 return cast<tensor::ParallelInsertSliceOp>(&insertOp)
                            .getDest()
                            .getDefiningOp() == op;
               });
    if (isDestination) continue;

    opsToClone.push_back(op);

    // Add all operands to the worklist.
    for (Value operand : op->getOperands()) {
      Operation *operandOp = operand.getDefiningOp();
      if (!operandOp) continue;
      worklist.insert(operandOp);
    }
  }

  // 2. Clone ops and replace their uses inside the ForeachThreadOp.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(
      &foreachThreadOp.getRegion().getBlocks().front());
  for (Operation *op : llvm::reverse(opsToClone)) {
    Operation *cloned = rewriter.clone(*op);
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : op->getUses())
      if (foreachThreadOp->isProperAncestor(use.getOwner()))
        uses.push_back(&use);
    for (OpOperand *use : uses) {
      unsigned resultNum = use->get().cast<OpResult>().getResultNumber();
      rewriter.updateRootInPlace(
          use->getOwner(), [&]() { use->set(cloned->getOpResult(resultNum)); });
    }
  }
}

/// Rewrite a ForeachThreadOp into a Flow::DispatchWorkGroupsOp.
/// This rewrite proceeds in a few steps:
///   - Step 0: Clone certain ops into the ForeachThreadOp (as per IREE
///     heuristic), so that they are part of the dispatch region.
///   - Step 1: Compute the result types and their result dynamic dim operands.
///     This first step takes advantage of the ops contained in the
///     ForeachThreadOp terminator and that are tied to the results.
///   - Step 2: Get values defined above and separate them between non-tensors,
///     tensors and introduce appropriate tensor dims.
///   - Step 3: Create ordered vectors of operands to pass to the builder and
///     build the dispatchOp.
///   - Step 4: Populate the workgroupCount region of the dispatchOp and set
///     the workload operands to the values defined above.
///   - Step 5: Fixup dispatchOp bbArgs and terminator.
///   - Step 6: Move the body of foreachThreadOp to the dispatchOp.
///   - Step 7: Set up bvm for RAUWIf. In particular, tensor operands become
///     flow dispatch tensor bbArgs and need to be
///     flow.dispatch.tensor.load'ed.
///   - Step 8: Plug dispatch workgroup id and count values into the bvm.
///   - Step 9. Rewrite tensor::ExtractSlice and ParallelInsert ops to the
///     relevant Flow DispatchTensorLoad/Store version.
///   - Step 10: Perform RAUWIf.
///   - Step 11: Drop the terminator and replace foreachThreadOp.
// TODO: n-D ForeachThreadOp
FailureOr<Flow::DispatchWorkgroupsOp>
rewriteForeachThreadToFlowDispatchWorkgroups(
    scf::ForeachThreadOp foreachThreadOp, PatternRewriter &rewriter) {
  // Step 0: Clone ops into the ForeachThreadOp.
  cloneOpsIntoForeachThreadOp(rewriter, foreachThreadOp);

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(foreachThreadOp);

  // Entry point start just before the foreachThreadOp.
  Location loc = foreachThreadOp.getLoc();
  scf::PerformConcurrentlyOp performConcurrentlyOp =
      foreachThreadOp.getTerminator();

  // Step 1: Compute all dynamic result dims.
  // The `dest` of the ParallelInsertSliceOp are tied to the results and carry
  // over to the Flow::DispatchWorkgroupsOp.
  // Use a SetVector to ensure tensor operand uniqueness.
  llvm::SetVector<Value> resultTensorOperands, resultTensorsDynamicDims;
  for (const Operation &yieldingOp : performConcurrentlyOp.getYieldingOps()) {
    auto parallelInsertOp = cast<tensor::ParallelInsertSliceOp>(&yieldingOp);
    Value dest = parallelInsertOp.getDest();
    bool inserted = resultTensorOperands.insert(dest);
    if (!inserted) continue;
    auto dynamicDims =
        getIndicesOfDynamicDims(dest.getType().cast<ShapedType>());
    for (int64_t dim : dynamicDims)
      resultTensorsDynamicDims.insert(
          rewriter.create<tensor::DimOp>(loc, dest, dim));
  }
  assert(resultTensorOperands.size() == foreachThreadOp.getNumResults() &&
         "Expected as many resultTensorOperands as results of foreachThreadOp");

  // Step 2. Get values defined above and separate them between non-tensors,
  // tensors and introduce appropriate tensor dims.
  // Tensors that have already been recorded as resultTensorOperands are
  // omitted to avoid duplications.
  llvm::SetVector<Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(foreachThreadOp.getRegion(),
                                  valuesDefinedAbove);

  SmallVector<Value> nonTensorOperands, tensorOperands, tensorDynamicDims;
  for (Value v : valuesDefinedAbove) {
    auto tensorType = v.getType().dyn_cast<RankedTensorType>();
    if (!tensorType) {
      nonTensorOperands.push_back(v);
      continue;
    }
    if (resultTensorOperands.contains(v)) continue;
    tensorOperands.push_back(v);
    for (int64_t dim : getIndicesOfDynamicDims(tensorType))
      tensorDynamicDims.push_back(rewriter.create<tensor::DimOp>(loc, v, dim));
  }

  // Step 3. Create ordered vectors of operands to pass to the builder and
  // build the dispatchOp. The dispatchOp is created with an empty
  // workgroup_count region and empty workload. They are populated next
  SmallVector<Value> nonDimOperands;
  llvm::append_range(nonDimOperands, nonTensorOperands);
  llvm::append_range(nonDimOperands, tensorOperands);
  llvm::append_range(nonDimOperands, resultTensorOperands);
  // scf::ForeachThreadOp tensors inserted into are tied to results and
  // translate to the tied operands of the dispatch.
  int64_t sizeNonTensors = nonTensorOperands.size();
  int64_t sizeNonResultTensors = tensorOperands.size();
  int64_t sizeResultTensors = resultTensorOperands.size();
  auto tiedOperandsSequence = llvm::seq<int64_t>(
      sizeNonTensors + sizeNonResultTensors,
      sizeNonTensors + sizeNonResultTensors + sizeResultTensors);
  // Separate out tensorOperands and tensorDynamicDims for RAUWIf purposes.
  SmallVector<Value> allTensorOperands = tensorOperands;
  llvm::append_range(allTensorOperands, resultTensorOperands);
  SmallVector<Value> allTensorDynamicDims = tensorDynamicDims;
  llvm::append_range(allTensorDynamicDims, resultTensorsDynamicDims);
  // clang-format off
  auto dispatchOp = rewriter.create<Flow::DispatchWorkgroupsOp>(
      loc,
      /*workload=*/ValueRange{},
      /*resultTypes=*/foreachThreadOp.getResultTypes(),
      /*resultDims=*/resultTensorsDynamicDims.getArrayRef(),
      /*operands=*/nonDimOperands,
      /*operandDims=*/allTensorDynamicDims,
      /*tiedOperands=*/llvm::to_vector<4>(tiedOperandsSequence));
  // clang-format on

  // Step 4. Outline the compute workload region and set up the workload
  // operands.
  if (failed(populateWorkgroupCountComputingRegion(rewriter, foreachThreadOp,
                                                   dispatchOp)))
    return foreachThreadOp->emitOpError(
               "failed to populate workload region for dispatchOp: ")
           << dispatchOp;

  // Step 5. Fixup dispatchOp bbArgs and terminator.
  // TODO: Ideally the builder would have created the proper bbArgs and the
  // ceremonial terminator.
  // Atm, the bbArgs for the region are missing index entries for the dynamic
  // dims of the tensor operands
  // We add them, following this convention:
  //  - The first `sizeNonTensors` bbArgs correspond to the non-tensor operands.
  //    These are already added by the builder and we leave them alone.
  //  - The next `sizeNonResultTensors + sizeResultTensors` bbArgs correspond to
  //    the tensor operands (non-result tensors followed by result tensors).
  //    These are already added by the builder and we leave them alone.
  //  - The next `tensorDynamicDims.size() + resultTensorsDynamicDims.size()`
  //    bbArgs correspond to the dynamic dimensions of the tensor operands and
  //    tensor results.
  //    These are *not yet* added by the builder and we add them explicitly.
  //    These index bbArgs are added after all tensor bbArgs and become the
  //    trailing bbArgs.
  //    Another possibility would be to interleave (tensor, tensor dynamic
  //    dims) but unless proven wrong, the trailing indices convention is
  //    quite simpler to implement: if bugs surface, these should be fixed or
  //    a real convention + verification should be adopted on the op + builder.
  Region &region = dispatchOp.getWorkgroupBody();
  Block *block = &region.front();
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToEnd(block);
    rewriter.create<Flow::ReturnOp>(loc);
  }
  // Add trailing index bbArgs and perform a basic sanity check.
  block->addArguments(
      SmallVector<Type>(allTensorDynamicDims.size(), rewriter.getIndexType()),
      SmallVector<Location>(allTensorDynamicDims.size(), loc));
  SmallVector<Value> allOperands = nonDimOperands;
  llvm::append_range(allOperands, allTensorDynamicDims);
  assert(block->getNumArguments() == allOperands.size() &&
         "Expected as many bbArgs as operands");

  // Step 6. Move the body of foreachThreadOp to the dispatchOp.
  block->getOperations().splice(
      block->begin(), foreachThreadOp.getRegion().front().getOperations());

  // Step 7. Set up bvm for RAUWIf.
  // Generally, allOperands map to their corresponding bbArg but there is a
  // twist: tensor operands map to flow.dispatch.tensor bbArgs and we need to
  // insert an explicit Flow::DispatchTensorLoadOp to get back a proper
  // tensor. Save the tensor operand -> flow tensor bbArg mapping in
  // `tensorToFlowBvm`.
  BlockAndValueMapping bvm, tensorToFlowBvm;
  auto flowBbArgs = block->getArguments().slice(
      sizeNonTensors, sizeNonResultTensors + sizeResultTensors);
  tensorToFlowBvm.map(allTensorOperands, flowBbArgs);
  assert(allOperands.size() == block->getArguments().size() &&
         "expected same number of operands and bbArgs");
  bvm.map(allOperands, block->getArguments());
  auto allTensorDimsBBArgs = block->getArguments().slice(
      nonDimOperands.size(), allTensorDynamicDims.size());
  for (auto en : llvm::enumerate(allTensorOperands)) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(block);
    // Warning: findVariadicDynamicDims needs to use the RankedTensorTypes and
    // does not work out of the box with Flow::DispatchTensorType.
    auto dynamicDims = Util::findVariadicDynamicDims(
        en.index(), allTensorOperands, allTensorDimsBBArgs);
    auto loadOp = rewriter.create<Flow::DispatchTensorLoadOp>(
        loc, en.value().getType().cast<RankedTensorType>(),
        tensorToFlowBvm.lookup(en.value()), dynamicDims);
    // Replace the tensor -> flow.dispatch.tensor entry by a
    // tensor -> flow.dispatch.tensor.load entry.
    bvm.map(en.value(), loadOp.getResult());
  }

  // Step 8. Plug dispatch workgroup id and count values into the bvm.
  rewriter.setInsertionPointToStart(block);
  SmallVector<Value, 8> workgroupIds, workgroupCounts;
  for (int64_t rank :
       llvm::seq<int64_t>(0, foreachThreadOp.getThreadIndices().size())) {
    workgroupIds.push_back(
        rewriter.create<Flow::DispatchWorkgroupIDOp>(loc, rank));
    workgroupCounts.push_back(
        rewriter.create<Flow::DispatchWorkgroupCountOp>(loc, rank));
  }
  bvm.map(foreachThreadOp.getThreadIndices(), workgroupIds);
  bvm.map(foreachThreadOp.getNumThreads(), workgroupCounts);

  // Step 9. Rewrite tensor::ExtractSlice and ParallelInsert ops to the
  // relevant Flow DispatchTensorLoad/Store version.
  rewriteParallelInsertSlices(rewriter, performConcurrentlyOp, *block,
                              resultTensorOperands.getArrayRef(),
                              resultTensorsDynamicDims.getArrayRef(),
                              tensorToFlowBvm);
  rewriteExtractSlices(rewriter, dispatchOp, allTensorOperands,
                       allTensorDynamicDims, tensorToFlowBvm);

  // Step 10. Perform RAUWIf.
  for (auto mapEntry : bvm.getValueMap()) {
    assert(mapEntry.first.getType() == mapEntry.second.getType() &&
           "must have the same type");
    mapEntry.first.replaceUsesWithIf(mapEntry.second, [&](OpOperand &use) {
      return dispatchOp->isProperAncestor(use.getOwner());
    });
  }

  // Step 11. Drop the terminator and replace foreachThreadOp.
  rewriter.eraseOp(performConcurrentlyOp);
  rewriter.replaceOp(foreachThreadOp, dispatchOp.getResults());

  return dispatchOp;
}

DiagnosedSilenceableFailure
transform_dialect::ForeachThreadToFlowDispatchWorkgroupsOp::applyToOne(
    scf::ForeachThreadOp target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  SimplePatternRewriter rewriter(target->getContext());
  FailureOr<Flow::DispatchWorkgroupsOp> result =
      rewriteForeachThreadToFlowDispatchWorkgroups(target, rewriter);
  if (failed(result))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  results.push_back(*result);
  return DiagnosedSilenceableFailure(success());
}

/// Return `true` if the given type is a ShapedType and has at least one
/// dynamic dimension.
static bool hasDynamicShape(Type t) {
  auto shapedType = t.dyn_cast<ShapedType>();
  if (!shapedType) return false;
  return !shapedType.hasStaticShape();
}

/// Reify the dynamic dimensions of the given value.
static LogicalResult reifyDynamicResultDims(OpBuilder b, Value value,
                                            SmallVector<Value> &dynamicDims) {
  OpBuilder::InsertionGuard guard(b);

  // Case 1: No dynamic result dims.
  if (!hasDynamicShape(value.getType())) return success();

  // There is at least one dynamic dimension, continue...
  ShapedType shapedType = value.getType().cast<ShapedType>();

  // Case 2: Value is a block argument.
  if (auto bbArg = value.dyn_cast<BlockArgument>()) {
    b.setInsertionPointToStart(bbArg.getOwner());
    for (int64_t i = 0; i < shapedType.getRank(); ++i) {
      if (shapedType.isDynamicDim(i)) {
        Value dim = b.create<tensor::DimOp>(bbArg.getLoc(), bbArg, i);
        dynamicDims.push_back(dim);
      }
    }
    return success();
  }

  // Value is an OpResult.
  Operation *op = value.getDefiningOp();
  OpResult opResult = value.cast<OpResult>();
  b.setInsertionPoint(op);

  // Case 3: Value is tied. Reify the dimensions of the tied operand.
  auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op);
  if (tiedOp) {
    Value tiedOperand = tiedOp.getTiedResultOperand(value);
    if (tiedOperand) {
#ifndef NDEBUG
      ShapedType tiedOperandType = tiedOperand.getType().cast<ShapedType>();
      assert(tiedOperandType == shapedType && "expected same type");
#endif  // NDEBUG
      return reifyDynamicResultDims(b, tiedOperand, dynamicDims);
    }
  }

  // Case 4: Query ShapeAwareOpInterface.
  auto shapeAwareOp = dyn_cast<IREE::Util::ShapeAwareOpInterface>(op);
  if (shapeAwareOp) {
    ValueRange dims =
        shapeAwareOp.getResultDynamicDims(opResult.getResultNumber());
    dynamicDims.append(dims.begin(), dims.end());
    return success();
  }

  // Case 5: Query ReifyRankedShapedTypeOpInterface.
  auto reifyShapeOp = dyn_cast<ReifyRankedShapedTypeOpInterface>(op);
  if (reifyShapeOp) {
    ReifiedRankedShapedTypeDims dims;
    if (failed(reifyShapeOp.reifyResultShapes(b, dims))) return failure();
    for (int64_t i = 0; i < shapedType.getRank(); ++i)
      if (shapedType.isDynamicDim(i))
        dynamicDims.push_back(dims[opResult.getResultNumber()][i]);
    return success();
  }

  return failure();
}

/// Reify the dynamic dimensions of all results of the given op.
static LogicalResult reifyDynamicResultDims(OpBuilder b, Operation *op,
                                            SmallVector<Value> &dynamicDims) {
  for (Value v : op->getResults())
    if (failed(reifyDynamicResultDims(b, v, dynamicDims))) return failure();

  return success();
}

static LogicalResult moveIntoDispatchRegion(RewriterBase &rewriter,
                                            Operation *target,
                                            Flow::DispatchRegionOp &regionOp) {
  OpBuilder::InsertionGuard guard(rewriter);
  Block &body = regionOp.getBody().front();
  DominanceInfo domInfo;

  // Case 1: `target` dominates `regionOp`.
  //
  // Example:
  //
  // %0 = "some_op"() : () -> (tensor<?xf32>)
  // %r = flow.dispatch.region -> (tensor<?xf32>{%d0}) {
  //   %1 = "another_op"(%0) : (tensor<?xf32>) -> (tensor<?xf32>)
  //   flow.return %1 : tensor<?xf32>
  // }
  // %2 = "yet_another_use"(%0) : (tensor<?xf32>) -> (tensor<?xf32>)
  //
  // In this example, "some_op" will be cloned into the dispatch region and the
  // OpOperand of "another_op" will be replaced:
  //
  // %0 = "some_op"() : () -> (tensor<?xf32>)
  // %r = flow.dispatch.region -> (tensor<?xf32>{%d0}) {
  //   %0_clone = "some_op"() : () -> (tensor<?xf32>)
  //   %1 = "another_op"(%0_clone) : (tensor<?xf32>) -> (tensor<?xf32>)
  //   flow.return %1 : tensor<?xf32>
  // }
  // %2 = "yet_another_use"(%0) : (tensor<?xf32>) -> (tensor<?xf32>)
  //
  // Note: If it weren't for "yet_another_use", the original "some_op" could
  // fold away.
  //
  // Note: If there were no "another_op" in the dispatch region, the IR would
  // not be modified and this function would fail.
  if (domInfo.properlyDominates(target, regionOp)) {
    SmallVector<OpOperand *> consumers;
    for (OpOperand &use : target->getUses())
      if (regionOp->isProperAncestor(use.getOwner())) consumers.push_back(&use);
    // We currently only fuse the consumers inside the regionOp. If there are
    // no consumers, the op will not be moved inside the regionOp. This could
    // be customized with a flag, such that target is always cloned into the
    // regionOp and uses that post-dominate the regionOp are updated to the a
    // newly added regionOp result.
    // E.g., in the above example, flow.return would also return %0_clone and
    // "yet_another_use" would use %r#1 instead of %0.
    if (!consumers.empty()) {
      rewriter.setInsertionPointToStart(&body);
      Operation *cloned = rewriter.clone(*target);
      for (OpOperand *consumer : consumers) {
        rewriter.updateRootInPlace(consumer->getOwner(), [&]() {
          consumer->set(cloned->getResult(
              consumer->get().cast<OpResult>().getResultNumber()));
        });
      }

      // Delete op if it no longer has any uses.
      if (target->use_empty()) rewriter.eraseOp(target);

      return success();
    }
  }

  // Case 2: `target` post-dominates `regionOp`. The target is cloned into the
  // regionOp and all original uses of the target are replaced with a newly
  // added regionOp result.
  //
  // Example:
  //
  // %r = flow.dispatch.region -> (tensor<?xf32>{%d}) {
  //   %0 = ...
  //   flow.return %0 : tensor<?xf32>
  // }
  // %1 = "some_op"(%r) : (tensor<?xf32>) -> (tensor<?xf32>)
  // "some_use"(%1) : (tensor<?xf32>) -> ()
  //
  // In this example, "some_op" will be cloned into the dispatch region and all
  // of its uses are replaced.
  //
  // %r:2 = flow.dispatch.region -> (tensor<?xf32>{%d}, tensor<?xf32>{%e}) {
  //   %0 = ...
  //   %1_clone = "some_op"(%0) : (tensor<?xf32>) -> (tensor<?xf32>)
  //   flow.return %0, %1_clone : tensor<?xf32>, tensor<?xf32>
  // }
  // "some_use"(%r#1) : (tensor<?xf32>) -> ()
  //
  // Note: If there is no use of %r#0, the dispatch region can canonicalize to
  // an op with only one result.
  if (!domInfo.properlyDominates(regionOp, target)) return failure();

  // This transform works only if all operands of the target op dominate (or are
  // equal to) the regionOp.
  bool usesDominateRegionOp = true;
  for (OpOperand &use : target->getOpOperands()) {
    auto opResult = use.get().dyn_cast<OpResult>();
    if (!opResult) continue;
    if (!domInfo.dominates(opResult.getDefiningOp(), regionOp))
      usesDominateRegionOp = false;
  }
  if (usesDominateRegionOp) {
    rewriter.setInsertionPoint(regionOp);
    // Determine dynamic result dims.
    SmallVector<Value> dynamicDims(regionOp.getResultDims().begin(),
                                   regionOp.getResultDims().end());
    if (failed(reifyDynamicResultDims(rewriter, target, dynamicDims)))
      return failure();
    // Determine result types of new RegionOp.
    SmallVector<Type> resultTypes(regionOp.getResultTypes().begin(),
                                  regionOp.getResultTypes().end());
    resultTypes.append(target->getResultTypes().begin(),
                       target->getResultTypes().end());
    // Create RegionOp and move op inside.
    auto newRegionOp = rewriter.create<Flow::DispatchRegionOp>(
        regionOp->getLoc(), resultTypes, dynamicDims);
    newRegionOp.getBody().takeBody(regionOp.getBody());
    rewriter.replaceOp(regionOp, newRegionOp.getResults().take_front(
                                     regionOp->getNumResults()));
    // Clone op inside and update terminator.
    Flow::ReturnOp returnOp =
        cast<Flow::ReturnOp>(newRegionOp.getBody().front().getTerminator());
    SmallVector<Value> returnedValues(returnOp.getOperands().begin(),
                                      returnOp.getOperands().end());
    unsigned newResultsOffset = returnedValues.size();
    rewriter.setInsertionPoint(returnOp);
    Operation *cloned = rewriter.clone(*target);
    returnedValues.append(cloned->getResults().begin(),
                          cloned->getResults().end());
    rewriter.updateRootInPlace(
        returnOp, [&]() { returnOp.operandsMutable().assign(returnedValues); });
    // Update operands of the cloned op.
    rewriter.updateRootInPlace(cloned, [&]() {
      for (OpOperand &use : cloned->getOpOperands()) {
        Operation *producer = use.get().getDefiningOp();
        if (producer != newRegionOp) continue;
        use.set(
            returnOp
                .getOperands()[use.get().cast<OpResult>().getResultNumber()]);
      }
    });
    // Update uses of the old op.
    rewriter.replaceOp(target,
                       newRegionOp->getResults().drop_front(newResultsOffset));
    regionOp = newRegionOp;
    return success();
  }

  return failure();
}

DiagnosedSilenceableFailure transform_dialect::MoveIntoDispatchRegionOp::apply(
    transform::TransformResults &transformResults,
    transform::TransformState &state) {
  ArrayRef<Operation *> targetOps = state.getPayloadOps(getTarget());
  ArrayRef<Operation *> dispatchRegion =
      state.getPayloadOps(getDispatchRegion());

  // TODO: Multiple targetOps could be allowed.
  if (targetOps.size() != 1 || dispatchRegion.size() != 1)
    return DiagnosedSilenceableFailure(this->emitOpError(
        "requires exactly one target/dispatch region handle"));

  auto regionOp = dyn_cast<Flow::DispatchRegionOp>(dispatchRegion.front());
  if (!regionOp)
    return DiagnosedSilenceableFailure(
        this->emitOpError("expected 'dispatch.region' operand"));

  IRRewriter rewriter(regionOp->getContext());
  if (failed(moveIntoDispatchRegion(rewriter, targetOps.front(), regionOp)))
    return DiagnosedSilenceableFailure(
        reportUnknownTransformError(targetOps.front()));

  transformResults.set(getTransformed().cast<OpResult>(),
                       regionOp.getOperation());
  return DiagnosedSilenceableFailure(success());
}

static Flow::DispatchRegionOp makeEmptyDispatchRegion(RewriterBase &rewriter,
                                                      Location loc) {
  OpBuilder::InsertionGuard guard(rewriter);

  // Create RegionOp.
  auto regionOp = rewriter.create<Flow::DispatchRegionOp>(
      loc, /*resultTypes=*/TypeRange(), /*dynamicDims=*/ValueRange());
  Block &body = regionOp.getBody().emplaceBlock();
  rewriter.setInsertionPointToStart(&body);
  rewriter.create<Flow::ReturnOp>(loc, ValueRange());

  return regionOp;
}

static FailureOr<Flow::DispatchRegionOp> wrapInDispatchRegion(
    RewriterBase &rewriter, Operation *op) {
  OpBuilder::InsertionGuard guard(rewriter);
  // Make an empty dispatch region right before the op.
  rewriter.setInsertionPoint(op);
  Flow::DispatchRegionOp regionOp =
      makeEmptyDispatchRegion(rewriter, op->getLoc());
  // Move the op into the dispatch region.
  if (failed(moveIntoDispatchRegion(rewriter, op, regionOp))) return failure();
  return regionOp;
}

DiagnosedSilenceableFailure
transform_dialect::WrapInDispatchRegionOp::applyToOne(
    Operation *target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  IRRewriter rewriter(target->getContext());
  auto maybeRegionOp = wrapInDispatchRegion(rewriter, target);
  if (failed(maybeRegionOp))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  results.push_back(*maybeRegionOp);
  return DiagnosedSilenceableFailure(success());
}

static LogicalResult fuseConsumers(RewriterBase &rewriter,
                                   Flow::DispatchRegionOp &regionOp) {
  while (regionOp->hasOneUse()) {
    OpOperand &use = *regionOp->getUses().begin();
    Operation *consumer = use.getOwner();
    if (!isa<linalg::LinalgOp>(consumer)) break;
    // TODO: IREE::Flow::areLinalgOpsFusableUsingTileAndFuse
    if (failed(moveIntoDispatchRegion(rewriter, consumer, regionOp))) break;
  }
  return success();
}

static SmallVector<Operation *> getOpsInReverse(Block &block) {
  return llvm::to_vector(
      llvm::reverse(llvm::map_range(block, [](Operation &op) { return &op; })));
}

/// A rewriter that keeps track of ops that were deleted.
class TrackingRewriter : public IRRewriter {
 public:
  using IRRewriter::IRRewriter;

  bool wasOpDeleted(Operation *op) { return deletedOps.contains(op); }

 protected:
  void notifyOperationRemoved(Operation *op) override { deletedOps.insert(op); }

 private:
  llvm::DenseSet<Operation *> deletedOps;
};

static LogicalResult heuristicStage1(
    TrackingRewriter &rewriter, Block &body,
    SmallVector<Flow::DispatchRegionOp> &dispatchRegions) {
  // Create dispatch regions for certain LinalgOps and tilable ops.
  auto isRootOp = [](Operation *op) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      if (isa<linalg::GenericOp>(op))
        return linalgOp.getNumReductionLoops() != 0;
      return !isa<linalg::FillOp>(op);
    }
    return isa<TilingInterface>(op);
  };

  SmallVector<Operation *> worklist = getOpsInReverse(body);
  for (Operation *op : worklist) {
    if (rewriter.wasOpDeleted(op)) continue;
    if (op->getParentOfType<Flow::DispatchRegionOp>()) continue;
    if (op->use_empty()) continue;
    if (!isRootOp(op)) continue;

    // TODO: Use DestinationStyleInterface when available upstream.
    linalg::OpOperandVector outOperands =
        TypeSwitch<Operation *, linalg::OpOperandVector>(op)
            .Case<linalg::LinalgOp>([&](auto linalgOp) {
              return linalgOp.getOutputTensorOperands();
            })
            .Default(
                [&](Operation *) -> linalg::OpOperandVector { return {}; });

    // Create a new dispatch region for `op`.
    auto maybeRegionOp = wrapInDispatchRegion(rewriter, op);
    if (failed(maybeRegionOp)) return failure();
    Flow::DispatchRegionOp regionOp = *maybeRegionOp;

    // Fuse producers into the dispatch region.
    for (OpOperand *operand : outOperands) {
      // Currently only fuse with producer ops that are `LinalgOp`s.
      auto producer = operand->get().getDefiningOp<linalg::LinalgOp>();
      if (!producer) continue;

      // Fuse with the consumer if all uses of producer are dominated by it.
      // TODO: clEnableMultiResultDispatches
      if (!producer->hasOneUse()) continue;
      if (producer.getNumLoops() != producer.getNumParallelLoops()) continue;

      if (failed(moveIntoDispatchRegion(rewriter, producer, regionOp)))
        return failure();
    }

    // Fuse consumers into the dispatch region.
    if (failed(fuseConsumers(rewriter, regionOp))) return failure();

    dispatchRegions.push_back(regionOp);
  }

  return success();
}

static LogicalResult heuristicStage2(
    TrackingRewriter &rewriter, Block &body,
    SmallVector<Flow::DispatchRegionOp> &dispatchRegions) {
  SmallVector<Operation *> worklist = getOpsInReverse(body);
  for (Operation *op : worklist) {
    if (rewriter.wasOpDeleted(op)) continue;
    // Target remaining LinalgOps that are not FillOps.
    if (!isa<linalg::LinalgOp>(op) || isa<linalg::FillOp>(op)) continue;

    // Create a new dispatch region for `op`.
    auto maybeRegionOp = wrapInDispatchRegion(rewriter, op);
    if (failed(maybeRegionOp)) return failure();
    Flow::DispatchRegionOp regionOp = *maybeRegionOp;

    // Fuse consumers into the dispatch region.
    if (failed(fuseConsumers(rewriter, regionOp))) return failure();

    dispatchRegions.push_back(regionOp);
  }

  return success();
}

static LogicalResult heuristicStage3(
    TrackingRewriter &rewriter, Block &body,
    SmallVector<Flow::DispatchRegionOp> &dispatchRegions) {
  // For each dispatch region: Clone certain ops into the body.
  for (Flow::DispatchRegionOp regionOp : dispatchRegions) {
    // Find captured SSA values from parent blocks.
    llvm::SetVector<Value> valuesDefinedAbove;
    mlir::getUsedValuesDefinedAbove(regionOp.getBody(), valuesDefinedAbove);
    SmallVector<Operation *> worklist;
    for (Value v : valuesDefinedAbove)
      if (Operation *op = v.getDefiningOp()) worklist.push_back(op);

    // Check each op.
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (rewriter.wasOpDeleted(op)) continue;
      if (!isa<arith::IndexCastOp, linalg::InitTensorOp, tensor::CastOp,
               tensor::ExtractOp, tensor::ExtractSliceOp, tensor::PadOp,
               arith::ConstantOp>(op))
        continue;
      if (failed(moveIntoDispatchRegion(rewriter, op, regionOp)))
        return failure();
      for (Value v : op->getOperands())
        if (Operation *op = v.getDefiningOp()) worklist.push_back(op);
    }
  }
  return success();
}

static FailureOr<SmallVector<Flow::DispatchRegionOp>> makeDispatchRegions(
    Block &body) {
  TrackingRewriter rewriter(body.getParentOp()->getContext());
  SmallVector<Flow::DispatchRegionOp> dispatchRegions;

  if (failed(heuristicStage1(rewriter, body, dispatchRegions)))
    return failure();

  if (failed(heuristicStage2(rewriter, body, dispatchRegions)))
    return failure();

  if (failed(heuristicStage3(rewriter, body, dispatchRegions)))
    return failure();

  return dispatchRegions;
}

DiagnosedSilenceableFailure transform_dialect::MakeDispatchRegionsOp::apply(
    transform::TransformResults &transformResults,
    transform::TransformState &state) {
  ArrayRef<Operation *> targetOps = state.getPayloadOps(getTarget());
  if (targetOps.size() != 1)
    return DiagnosedSilenceableFailure(
        this->emitOpError("requires exactly one target handle"));

  SmallVector<Operation *> dispatchRegions;
  auto moduleOp = cast<ModuleOp>(targetOps.front());
  auto walkResult = moduleOp.walk([&](FunctionOpInterface funcOp) {
    for (Block &block : funcOp.getBody()) {
      auto maybeRegionOps = makeDispatchRegions(block);
      if (failed(maybeRegionOps)) return WalkResult::interrupt();
      auto regionOps =
          llvm::map_range(*maybeRegionOps, [](Flow::DispatchRegionOp regionOp) {
            return regionOp.getOperation();
          });
      dispatchRegions.append(regionOps.begin(), regionOps.end());
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return DiagnosedSilenceableFailure(failure());

  transformResults.set(getDispatchRegion().cast<OpResult>(), dispatchRegions);
  return DiagnosedSilenceableFailure(success());
}

//===---------------------------------------------------------------------===//
// ContainingOpToFlowDispatchWorkgroupsOp
//===---------------------------------------------------------------------===//

/// Compute the dynamic dims of the given value and add them to the vector.
static void appendDynamicDims(OpBuilder &b, Location loc,
                              SmallVector<Value> &argumentDims, Value tensor) {
  auto tensorType = tensor.getType().cast<RankedTensorType>();

  // Fast-path for if the value comes from ops that support our dynamic
  // shape interfaces. Otherwise we have to insert tensor.dim ops.
  auto availableDims = IREE::Util::findDynamicDims(tensor);
  if (availableDims.hasValue()) {
    argumentDims.append(availableDims->begin(), availableDims->end());
    assert(tensorType.getNumDynamicDims() == availableDims->size() &&
           "not enough dynamic dims found");
    return;
  }

  for (auto dim : llvm::enumerate(tensorType.getShape())) {
    if (dim.value() != ShapedType::kDynamicSize) continue;
    argumentDims.push_back(
        b.createOrFold<tensor::DimOp>(loc, tensor, dim.index()));
  }
}

/// Follow the reverse SSA use-def chain of the given value (always taking the
/// tied operand) and return the first value outside of `regionOp`.
static Optional<Value> findFirstTiedValueOutsideOfRegionOp(
    Flow::DispatchRegionOp regionOp, Value value) {
  // Check if `v` is defined outside of `regionOp`.
  auto isOutside = [&](Value v) {
    if (v.isa<OpResult>()) return !regionOp->isAncestor(v.getDefiningOp());
    assert(v.isa<BlockArgument>() && "expected bbArg");
    // DispatchRegionOp does not have block arguments.
    return true;
  };

  while (!isOutside(value)) {
    auto tiedOpInterface = value.getDefiningOp<IREE::Util::TiedOpInterface>();
    if (!tiedOpInterface)
      // Reached an op that does not implement the interface.
      return llvm::None;
    value = tiedOpInterface.getTiedResultOperand(value);
    if (!value)
      // Nothing is tied here.
      return llvm::None;
  }

  return value;
}

/// Rewrite the DispatchRegionOp into a DispatchWorkgroupsOp. The
/// DispatchRegionOp is not isolated from above and may capture any SSA value
/// that is in scope. The generated DispatchWorkgroupsOp captures all SSA values
/// explicitly and makes them available inside the region via block arguments.
static FailureOr<Flow::DispatchWorkgroupsOp>
rewriteFlowDispatchRegionToFlowDispatchWorkgroups(
    Flow::DispatchRegionOp regionOp, RewriterBase &rewriter) {
  // Only ops with a single block are supported.
  Region &region = regionOp.getBody();
  if (!region.hasOneBlock()) return failure();
  Block &body = region.front();
  auto terminator = cast<Flow::ReturnOp>(body.getTerminator());
  unsigned numResults = terminator->getNumOperands();

  // Prepare rewriter.
  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = regionOp.getLoc();
  rewriter.setInsertionPoint(regionOp);

  // Compute arguments of the dispatch region.
  llvm::SetVector<Value> argumentsSet;
  mlir::getUsedValuesDefinedAbove(region, argumentsSet);
  // Unranked tensors are not supported.
  if (llvm::any_of(argumentsSet, [](Value v) {
        return v.getType().isa<UnrankedTensorType>();
      }))
    return failure();
  // Only ranked tensor results are supported.
  if (!llvm::all_of(regionOp.getResultTypes(),
                    [](Type t) { return t.isa<RankedTensorType>(); }))
    return failure();

  // Compute dimensions of tensor args.
  SmallVector<Value> argumentDims;
  for (Value tensor : argumentsSet) {
    auto tensorType = tensor.getType().dyn_cast<RankedTensorType>();
    if (!tensorType) continue;
    appendDynamicDims(rewriter, loc, argumentDims, tensor);
  }

  // Find tied results.
  DenseSet<Value> tiedArgumentsSet;
  SmallVector<int64_t> tiedArguments(numResults,
                                     IREE::Util::TiedOpInterface::kUntiedIndex);
  for (const auto &it : llvm::enumerate(terminator->getOperands())) {
    auto tiedArgument =
        findFirstTiedValueOutsideOfRegionOp(regionOp, it.value());
    if (!tiedArgument.hasValue()) continue;
    assert(argumentsSet.contains(*tiedArgument) &&
           "expected that tiedArgument is already an argument");
    // Do not tie an argument to multiple results.
    if (tiedArgumentsSet.contains(*tiedArgument)) continue;
    tiedArgumentsSet.insert(*tiedArgument);
    tiedArguments[it.index()] = std::distance(
        argumentsSet.begin(), llvm::find(argumentsSet, *tiedArgument));
  }

  // Create empty dispatch region.
  SmallVector<Value> arguments(argumentsSet.begin(), argumentsSet.end());
  arguments.append(argumentDims);
  for (unsigned i = 0; i < numResults; ++i) {
    // Tied arguments already have their dynamic result dims in `arguments`. Do
    // not add them again.
    if (tiedArguments[i] == IREE::Util::TiedOpInterface::kUntiedIndex) {
      ValueRange dims = regionOp.getResultDynamicDims(i);
      arguments.append(dims.begin(), dims.end());
    }
  }
  auto workgroupsOp = rewriter.create<IREE::Flow::DispatchWorkgroupsOp>(
      loc, /*workload=*/ValueRange(), regionOp.getResultTypes(),
      regionOp.getResultDims(), arguments, argumentDims, tiedArguments);
  BlockAndValueMapping bvm;
  bvm.map(arguments, workgroupsOp.getInputBlockArguments());

  // Create DispatchTensorLoadOp for all tensor arguments.
  assert(workgroupsOp.getWorkgroupBody().hasOneBlock() &&
         "expected one block after constructor");
  Block &newBody = workgroupsOp.getWorkgroupBody().getBlocks().front();
  assert(newBody.empty() && "expected empty block after constructor");
  rewriter.setInsertionPointToStart(&newBody);
  for (const auto &it : llvm::enumerate(arguments)) {
    auto tensorType = it.value().getType().dyn_cast<RankedTensorType>();
    if (!tensorType) continue;
    auto inputBbArg = workgroupsOp.getInputBlockArgument(it.index());
    auto dims =
        Util::findVariadicDynamicDims(it.index(), arguments, argumentDims);
    assert(dims.size() == tensorType.getNumDynamicDims() &&
           "dynamic dims not found among arguments");
    SmallVector<Value> bbArgDims = llvm::to_vector(
        llvm::map_range(dims, [&](Value v) { return bvm.lookup(v); }));
    Value loadedTensor = rewriter.create<IREE::Flow::DispatchTensorLoadOp>(
        loc, tensorType, inputBbArg, bbArgDims);
    bvm.map(it.value(), loadedTensor);
  }

  // Move regionOp body into the workgroupsOp.
  newBody.getOperations().splice(newBody.end(), body.getOperations());
  for (Value argument : arguments) {
    argument.replaceUsesWithIf(bvm.lookup(argument), [&](OpOperand &operand) {
      return workgroupsOp->isProperAncestor(operand.getOwner());
    });
  }

  // Update terminator.
  rewriter.setInsertionPoint(terminator);
  for (const auto &it : llvm::enumerate(terminator->getOperands())) {
    auto outputBbArg = workgroupsOp.getOutputBlockArgument(it.index());
    ValueRange dims;
    if (tiedArguments[it.index()] ==
        IREE::Util::TiedOpInterface::kUntiedIndex) {
      dims = regionOp.getResultDynamicDims(it.index());
    } else {
      // This assumes that the number of dynamic dims does not change when
      // following an SSA use-def chain of tied values.
      dims = Util::findVariadicDynamicDims(tiedArguments[it.index()], arguments,
                                           argumentDims);
    }
#ifndef NDEBUG
    auto tensorType = it.value().getType().cast<RankedTensorType>();
    assert(dims.size() == tensorType.getNumDynamicDims() &&
           "mismatching number of dynamic dims");
#endif  // NDEBUG
    SmallVector<Value> bbArgDims = llvm::to_vector(
        llvm::map_range(dims, [&](Value v) { return bvm.lookup(v); }));
    rewriter.create<IREE::Flow::DispatchTensorStoreOp>(loc, it.value(),
                                                       outputBbArg, bbArgDims);
  }

  // Delete the old terminator and create a new one.
  rewriter.create<IREE::Flow::ReturnOp>(loc);
  rewriter.eraseOp(terminator);

  rewriter.replaceOp(regionOp, workgroupsOp.getResults());
  return workgroupsOp;
}

DiagnosedSilenceableFailure
transform_dialect::FlowDispatchRegionToFlowDispatchWorkgroupsOp::applyToOne(
    Flow::DispatchRegionOp target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  IRRewriter rewriter(target->getContext());
  FailureOr<Flow::DispatchWorkgroupsOp> result =
      rewriteFlowDispatchRegionToFlowDispatchWorkgroups(target, rewriter);
  if (failed(result))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  results.push_back(*result);
  return DiagnosedSilenceableFailure(success());
}

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Flow/TransformExtensions/FlowExtensionsOps.cpp.inc"
