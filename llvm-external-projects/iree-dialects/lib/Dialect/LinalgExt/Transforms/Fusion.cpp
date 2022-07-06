// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::LinalgExt;

FailureOr<SmallVector<TilingInterface>>
mlir::iree_compiler::IREE::LinalgExt::LinalgExtFusionInContainingOpPattern::
    returningMatchAndRewrite(TilingInterface producerOp,
                             PatternRewriter &rewriter) const {
  if (producerOp->getNumResults() != 1)
    return failure();

  // Search the producer slices accessed within the containing operation.
  SmallVector<tensor::ExtractSliceOp> sliceOps;
  for (Operation *user : producerOp->getUsers()) {
    auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    if (!sliceOp)
      continue;
    if (sliceOp->getParentOp() != containingOp)
      continue;
    sliceOps.push_back(sliceOp);
  }
  // Check for a non-empty list of fusion opportunities.
  if (sliceOps.empty())
    return failure();

  SmallVector<Value> destinationOperands =
      producerOp.getDestinationOperands(rewriter);

  // Try to fuse the producer in-place of the tensor::ExtractSliceOps.
  SmallVector<TilingInterface> fusedOps;
  for (tensor::ExtractSliceOp sliceOp : sliceOps) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(sliceOp);

    // Tile the producer.
    FailureOr<Value> tiledProducer = producerOp.generateResultTileValue(
        rewriter, /*resultNumber=*/0, destinationOperands,
        sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(), true);
    if (failed(tiledProducer))
      return failure();
    fusedOps.push_back(cast<TilingInterface>(tiledProducer->getDefiningOp()));
  }

  // Replace the tensor::ExtractSliceOps.
  for (const auto &en : enumerate(sliceOps))
    rewriter.replaceOp(en.value(), fusedOps[en.index()]->getResult(0));
  return fusedOps;
}

static FailureOr<Value> fuseOperand(Value operand, PatternRewriter &rewriter) {
  // Ensure that the operand is a slice of a producer result.
  auto sliceOp = operand.getDefiningOp<tensor::ExtractSliceOp>();
  if (!sliceOp)
    return failure();
  auto producerOp = sliceOp.source().getDefiningOp<TilingInterface>();
  if (!producerOp || producerOp->getNumResults() != 1)
    return failure();

  SmallVector<Value> destinationOperands =
      producerOp.getDestinationOperands(rewriter);

  // Tile the producer.
  return producerOp.generateResultTileValue(
      rewriter, /*resultNumber=*/0, destinationOperands,
      sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(), true);
}

FailureOr<FusionResult> LinalgExtFusionPattern::returningMatchAndRewrite(
    TilingInterface consumerOp, PatternRewriter &rewriter) const {
  // Try to fuse the producers of all operands to fuse.
  SmallVector<TilingInterface> fusedOps;
  for (int64_t operandToFuse : operandsToFuse) {
    // Check the operand exists.
    if (operandToFuse >= consumerOp->getNumOperands())
      return failure();

    // Tile the producer.
    FailureOr<Value> tiledProducer =
        fuseOperand(consumerOp->getOperand(operandToFuse), rewriter);
    if (failed(tiledProducer))
      return failure();

    // Update the consumer in-place using the tiled producer.
    rewriter.updateRootInPlace(consumerOp, [&]() {
      consumerOp->setOperand(operandToFuse, *tiledProducer);
    });
    fusedOps.push_back(cast<TilingInterface>(tiledProducer->getDefiningOp()));
  }

  return FusionResult{consumerOp, fusedOps};
}

LogicalResult mlir::iree_compiler::IREE::LinalgExt::fuseAllOperands(
    TilingInterface consumerOp, PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(consumerOp);
  bool changed = false;

  for (OpOperand &opOperand : consumerOp->getOpOperands()) {
    // Tile the producer.
    FailureOr<Value> tiledProducer = fuseOperand(opOperand.get(), rewriter);
    if (failed(tiledProducer))
      continue;

    // Update the consumer in-place using the tiled producer.
    rewriter.updateRootInPlace(consumerOp,
                               [&]() { opOperand.set(*tiledProducer); });

    changed = true;
  }

  return success(changed);
}
