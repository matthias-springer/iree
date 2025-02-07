// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <numeric>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/IR/ClosureOpUtils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

//===----------------------------------------------------------------------===//
// Folding utilities
//===----------------------------------------------------------------------===//

// Returns a new set of dynamic dimensions for a shape carrying op when a type
// is being changed. This attempts to reuse the existing dimension values if
// they are available and will drop/insert new ones as required.
static SmallVector<Value, 4> refreshDimsOnTypeChange(
    Operation *op, Type oldType, Type newType, ValueRange oldDims,
    PatternRewriter &rewriter) {
  if (oldType == newType) return llvm::to_vector<4>(oldDims);

  // Build an expanded list of all the dims - constants will be nullptr.
  // This lets us map back the new types without worrying about whether some
  // subset become static or dynamic.
  auto oldShapedType = oldType.cast<ShapedType>();
  SmallVector<Value, 4> allOldDims(oldShapedType.getRank());
  for (unsigned i = 0; i < oldShapedType.getRank(); ++i) {
    if (oldShapedType.isDynamicDim(i)) {
      allOldDims[i] = oldDims.front();
      oldDims = oldDims.drop_front();
    }
  }

  auto newShapedType = newType.cast<ShapedType>();
  SmallVector<Value, 4> newDims;
  for (unsigned i = 0; i < newShapedType.getRank(); ++i) {
    if (newShapedType.isDynamicDim(i)) {
      auto oldValue = allOldDims[i];
      if (oldValue) {
        // Old value valid; reuse.
        newDims.push_back(oldValue);
      } else {
        // Dimension has changed to be dynamic; insert a constant to use.
        // This sometimes happens during folding of casts and usually is cleaned
        // up pretty quickly.
        newDims.push_back(rewriter.createOrFold<arith::ConstantIndexOp>(
            op->getLoc(), oldShapedType.getDimSize(i)));
      }
    }
  }
  return newDims;
}

//===----------------------------------------------------------------------===//
// flow.dispatch.workgroups
//===----------------------------------------------------------------------===//

void DispatchWorkgroupsOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<IREE::Util::ClosureOptimizationPattern<DispatchWorkgroupsOp>>(
      context);
}

//===----------------------------------------------------------------------===//
// flow.dispatch.tie_shape
//===----------------------------------------------------------------------===//

OpFoldResult DispatchTieShapeOp::fold(ArrayRef<Attribute> operands) {
  if (dynamic_dims().empty()) {
    return operand();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// flow.dispatch.tensor.load
//===----------------------------------------------------------------------===//

namespace {

// Updates the |dimValues| of |tensorValue| with dimensions inferred from IR.
// The dimension values may be derived values that are redundant with captured
// dimensions and by redirecting to the captured values we can simplify things.
// Returns true if the dims were changed.
static bool updateTensorOpDims(Operation *op, Value tensorValue,
                               MutableOperandRange mutableDimValues) {
  auto dynamicDimsOr = IREE::Util::findDynamicDims(tensorValue, op->getBlock(),
                                                   Block::iterator(op));
  if (!dynamicDimsOr.hasValue()) return false;
  auto dynamicDims = dynamicDimsOr.getValue();
  bool anyChanged = false;
  OperandRange oldValueRange = mutableDimValues;
  auto oldValues = llvm::to_vector<4>(oldValueRange);
  for (unsigned i = 0; i < dynamicDims.size(); ++i) {
    if (oldValues[i] != dynamicDims[i]) {
      mutableDimValues.slice(i, 1).assign(dynamicDims[i]);
      anyChanged = true;
    }
  }
  return anyChanged;
}

struct ReuseDispatchTensorLoadShapeDims
    : public OpRewritePattern<DispatchTensorLoadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DispatchTensorLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    return success(updateTensorOpDims(loadOp, loadOp.source(),
                                      loadOp.source_dimsMutable()));
  }
};

// Inlining producers of an input to the dispatch region results in the
// `flow.dispatch.input.load` having a `tensor` type as input. This fails
// verification. Since inlining happens during canonicalization, add a pattern
// to convert
//
// flow.dispatch.input.load %v, offsets .., sizes .., strides..
//   : tensor<...> -> tensor<..>
//
// to
//
// subtensor %v[..] [..] [..]
struct ConvertDispatchInputLoadOfTensorToSubTensor
    : public OpRewritePattern<DispatchTensorLoadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DispatchTensorLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    if (!loadOp.source().getType().isa<RankedTensorType>()) {
      return failure();
    }
    // If the offsets are empty rely on folding to take care of it.
    if (loadOp.offsets().empty() && loadOp.sizes().empty() &&
        loadOp.strides().empty()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        loadOp, loadOp.source(), loadOp.getMixedOffsets(),
        loadOp.getMixedSizes(), loadOp.getMixedStrides());
    return success();
  }
};

/// Pattern to rewrite a subview op with constant arguments.
struct DispatchTensorLoadOpWithOffsetSizesAndStridesConstantArgumentFolder final
    : public OpRewritePattern<DispatchTensorLoadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DispatchTensorLoadOp op,
                                PatternRewriter &rewriter) const override {
    // No constant operand, just return;
    if (llvm::none_of(op.getOperands(), [](Value operand) {
          return matchPattern(operand, matchConstantIndex());
        })) {
      return failure();
    }

    // At least one of offsets/sizes/strides is a new constant.
    // Form the new list of operands and constant attributes from the existing.
    SmallVector<OpFoldResult> mixedOffsets(op.getMixedOffsets());
    SmallVector<OpFoldResult> mixedSizes(op.getMixedSizes());
    SmallVector<OpFoldResult> mixedStrides(op.getMixedStrides());
    canonicalizeSubViewPart(mixedOffsets, ShapedType::isDynamicStrideOrOffset);
    canonicalizeSubViewPart(mixedSizes, ShapedType::isDynamic);
    canonicalizeSubViewPart(mixedStrides, ShapedType::isDynamicStrideOrOffset);

    // Create the new op in canonical form.
    auto resultType = DispatchTensorLoadOp::inferRankReducedResultType(
        op.result().getType().cast<RankedTensorType>().getRank(),
        op.source().getType().cast<DispatchTensorType>(), mixedSizes);
    auto newOp = rewriter.create<DispatchTensorLoadOp>(
        op.getLoc(), resultType, op.source(), op.source_dims(), mixedOffsets,
        mixedSizes, mixedStrides);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getResult().getType(),
                                                newOp.getResult());
    return success();
  }
};

}  // namespace

void DispatchTensorLoadOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ReuseDispatchTensorLoadShapeDims>(context);
  results.insert<ConvertDispatchInputLoadOfTensorToSubTensor>(context);
  results.insert<
      DispatchTensorLoadOpWithOffsetSizesAndStridesConstantArgumentFolder>(
      context);
}

// Inlining producers of an input to the dispatch region results in the
// `flow.dispatch.input.load` having a `tensor` type as input. This fails
// verification. Fold such uses of the offsets, size and strides are emtpy.
// i.e, flow.dispatch.input.load %v -> %v
OpFoldResult DispatchTensorLoadOp::fold(ArrayRef<Attribute> operands) {
  if (source().getType() && source().getType().isa<RankedTensorType>() &&
      getMixedOffsets().empty() && getMixedSizes().empty() &&
      getMixedStrides().empty()) {
    return source();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// flow.dispatch.tensor.store
//===----------------------------------------------------------------------===//

namespace {

struct ReuseDispatchTensorStoreShapeDims
    : public OpRewritePattern<DispatchTensorStoreOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    return success(updateTensorOpDims(storeOp, storeOp.target(),
                                      storeOp.target_dimsMutable()));
  }
};

struct FoldCastOpIntoDispatchStoreOp
    : public OpRewritePattern<DispatchTensorStoreOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    if (!storeOp.value().getDefiningOp<tensor::CastOp>()) return failure();
    auto parentOp = storeOp.value().getDefiningOp<tensor::CastOp>();
    rewriter.replaceOpWithNewOp<DispatchTensorStoreOp>(
        storeOp, parentOp.source(), storeOp.target(), storeOp.target_dims(),
        storeOp.offsets(), storeOp.sizes(), storeOp.strides(),
        storeOp.static_offsets(), storeOp.static_sizes(),
        storeOp.static_strides());
    return success();
  }
};

}  // namespace

void DispatchTensorStoreOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ReuseDispatchTensorStoreShapeDims>(context);
  results.insert<FoldCastOpIntoDispatchStoreOp>(context);
}

//===----------------------------------------------------------------------===//
// flow.dispatch.workgroup.*
//===----------------------------------------------------------------------===//

OpFoldResult DispatchWorkgroupRankOp::fold(ArrayRef<Attribute> operands) {
  if (auto dispatchOp = (*this)->getParentOfType<DispatchWorkgroupsOp>()) {
    return IntegerAttr::get(IndexType::get(getContext()),
                            APInt(64, dispatchOp.workgroup_count().size()));
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Tensor ops
//===----------------------------------------------------------------------===//

/// Reduces the provided multidimensional index into a flattended 1D row-major
/// index. The |type| is expected to be statically shaped (as all constants
/// are).
static uint64_t getFlattenedIndex(ShapedType type, ArrayRef<uint64_t> index) {
  assert(type.hasStaticShape() && "for use on statically shaped types only");
  auto rank = type.getRank();
  auto shape = type.getShape();
  uint64_t valueIndex = 0;
  uint64_t dimMultiplier = 1;
  for (int i = rank - 1; i >= 0; --i) {
    valueIndex += index[i] * dimMultiplier;
    dimMultiplier *= shape[i];
  }
  return valueIndex;
}

static bool compareShapesEqual(ShapedType lhsType, ValueRange lhsDynamicDims,
                               ShapedType rhsType, ValueRange rhsDynamicDims) {
  if (lhsType.hasStaticShape() && rhsType.hasStaticShape() &&
      lhsType == rhsType) {
    // Static shape equivalence means we can fast-path the check.
    return true;
  }
  if (lhsType.getRank() != rhsType.getRank()) {
    return false;
  }
  unsigned dynamicDimIndex = 0;
  for (unsigned i = 0; i < lhsType.getRank(); ++i) {
    if (lhsType.isDynamicDim(i) != rhsType.isDynamicDim(i)) {
      // Static/dynamic dimension mismatch - definitely differ.
      return false;
    } else if (lhsType.isDynamicDim(i)) {
      unsigned j = dynamicDimIndex++;
      if (lhsDynamicDims[j] != rhsDynamicDims[j]) {
        // Dynamic dimensions with different SSA values - probably differ.
        return false;
      }
    } else {
      if (lhsType.getDimSize(i) != rhsType.getDimSize(i)) {
        // Static dimensions differ.
        return false;
      }
    }
  }
  return true;
}

OpFoldResult TensorConstantOp::fold(ArrayRef<Attribute> operands) {
  auto dynamicType = getType();
  if (dynamicType.getNumDynamicDims() == 0) {
    return value();
  }
  return {};
}

namespace {

struct ExpandDynamicShapeConstant : public OpRewritePattern<TensorConstantOp> {
  using OpRewritePattern<TensorConstantOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto constantOp =
        rewriter.create<arith::ConstantOp>(op.getLoc(), op.value());
    auto dynamicType = op.getType();
    auto staticType = constantOp.getType().cast<ShapedType>();
    SmallVector<Value> dynamicDims;
    for (int64_t i = 0; i < dynamicType.getNumDynamicDims(); ++i) {
      auto dimValue = rewriter
                          .create<arith::ConstantIndexOp>(
                              op.getLoc(), staticType.getDimSize(i))
                          .getResult();
      dynamicDims.push_back(
          rewriter.create<IREE::Util::DoNotOptimizeOp>(op.getLoc(), dimValue)
              .getResult(0));
    }
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorReshapeOp>(
        op, dynamicType, constantOp.getResult(), dynamicDims);
    return success();
  }
};

}  // namespace

void TensorConstantOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ExpandDynamicShapeConstant>(context);
}

OpFoldResult TensorReshapeOp::fold(ArrayRef<Attribute> operands) {
  auto sourceType = source().getType().cast<ShapedType>();
  auto resultType = result().getType().cast<ShapedType>();
  if (compareShapesEqual(sourceType, source_dims(), resultType,
                         result_dims())) {
    // Shapes match and this is a no-op so just fold to the source.
    return source();
  }

  return {};
}

namespace {

// Flatten a chain of reshapes (reshape feeding into reshape) such that a
// reshape only ever pulls from a non-reshape source. This prevents big useless
// chains and makes it easier to track the original storage for the tensor.
struct FlattenTensorReshapeChain : public OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto sourceOp =
        dyn_cast_or_null<TensorReshapeOp>(reshapeOp.source().getDefiningOp());
    if (!sourceOp) return failure();

    // We want the same result value/shape but to source from the ancestor. We
    // need to pull any dynamic dims from that as we don't care about the
    // intermediate reshapes.
    rewriter.replaceOpWithNewOp<TensorReshapeOp>(
        reshapeOp, reshapeOp.result().getType(), sourceOp.source(),
        sourceOp.source_dims(), reshapeOp.result_dims());
    return success();
  }
};

// Replace `flow.tensor.splat`-`flow.tensor.load` op-pairs by the input
// primitive value for the splat op.
struct FoldSplatLoadIntoPrimitive : public OpRewritePattern<TensorLoadOp> {
  using OpRewritePattern<TensorLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto sourceOp =
        dyn_cast_or_null<TensorSplatOp>(loadOp.source().getDefiningOp());

    if (!sourceOp) return failure();

    rewriter.replaceOp(loadOp, sourceOp.value());
    return success();
  }
};

struct FoldSplatReshapeIntoSplat : public OpRewritePattern<TensorSplatOp> {
  using OpRewritePattern<TensorSplatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorSplatOp splatOp,
                                PatternRewriter &rewriter) const override {
    if (!splatOp.result().hasOneUse()) return failure();

    auto reshapeOp = dyn_cast_or_null<TensorReshapeOp>(
        splatOp.result().use_begin()->getOwner());
    if (!reshapeOp) return failure();

    rewriter.replaceOpWithNewOp<TensorSplatOp>(
        reshapeOp, reshapeOp.result().getType(), splatOp.value(),
        reshapeOp.result_dims());
    rewriter.eraseOp(splatOp);

    return success();
  }
};

struct ResolveShapedRank : public OpRewritePattern<tensor::RankOp> {
  using OpRewritePattern<tensor::RankOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::RankOp op,
                                PatternRewriter &rewriter) const override {
    auto shapedType = op.tensor().getType().cast<ShapedType>();
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op,
                                                        shapedType.getRank());
    return success();
  }
};

struct ResolveShapedDim : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getConstantIndex().hasValue()) {
      return rewriter.notifyMatchFailure(
          op, "non-constant index dim ops are unsupported");
    }
    auto idx = op.getConstantIndex().getValue();

    auto shapedType = op.source().getType().cast<ShapedType>();
    if (!shapedType.isDynamicDim(idx)) {
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(
          op, shapedType.getDimSize(idx));
      return success();
    }

    auto dynamicDims = IREE::Util::findDynamicDims(
        op.source(), op->getBlock(), Block::iterator(op.getOperation()));
    if (!dynamicDims.hasValue()) {
      return rewriter.notifyMatchFailure(op, "no dynamic dims found/usable");
    }
    unsigned dimOffset = 0;
    for (unsigned i = 0; i < idx; ++i) {
      if (shapedType.isDynamicDim(i)) ++dimOffset;
    }
    rewriter.replaceOp(op, dynamicDims.getValue()[dimOffset]);

    return success();
  }
};

}  // namespace

void TensorReshapeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<FlattenTensorReshapeChain>(context);
  results.insert<ResolveShapedRank>(context);
  results.insert<ResolveShapedDim>(context);
}

void TensorLoadOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<FoldSplatLoadIntoPrimitive>(context);
}

OpFoldResult TensorLoadOp::fold(ArrayRef<Attribute> operands) {
  if (auto source = operands[0].dyn_cast_or_null<ElementsAttr>()) {
    // Load directly from the constant source tensor.
    auto indices = operands.drop_front();
    if (llvm::count(indices, nullptr) == 0) {
      return source.getValues<Attribute>()[llvm::to_vector<4>(
          llvm::map_range(indices, [](Attribute value) {
            return value.cast<IntegerAttr>().getValue().getZExtValue();
          }))];
    }
  }
  return {};
}

OpFoldResult TensorStoreOp::fold(ArrayRef<Attribute> operands) {
  if (!operands[0]) return {};
  auto &value = operands[0];
  if (auto target = operands[1].dyn_cast_or_null<ElementsAttr>()) {
    // Store into the constant target tensor.
    if (target.getType().getRank() == 0) {
      return DenseElementsAttr::get(target.getType(), {value});
    }
    auto indices = operands.drop_front(2);
    if (llvm::count(indices, nullptr) == 0) {
      uint64_t offset = getFlattenedIndex(
          target.getType(),
          llvm::to_vector<4>(llvm::map_range(indices, [](Attribute value) {
            return value.cast<IntegerAttr>().getValue().getZExtValue();
          })));
      SmallVector<Attribute, 16> newContents(target.getValues<Attribute>());
      newContents[offset] = value;
      return DenseElementsAttr::get(target.getType(), newContents);
    }
  }
  return {};
}

void TensorSplatOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  // TODO(benvanik): canonicalize splat+slice to smaller splat.
  results.insert<FoldSplatReshapeIntoSplat>(context);
}

OpFoldResult TensorCloneOp::fold(ArrayRef<Attribute> operands) {
  if (operands[0]) {
    // Constants always fold.
    return operands[0];
  }

  // TODO(benvanik): elide clones when safe to do so. Right now clone is
  // load-bearing to work around our lack of cross-stream scheduling. Clones are
  // inserted to avoid mutating function arguments and any logic we perform here
  // (without *also* checking all the conditions that may insert a clone) will
  // just fight.
  //
  // Once the clones are not load-bearing we can remove them in all the normal
  // cases (one user, no intervening uses between clone and consumers of
  // operands, etc).

  return {};
}

// Slices tensor from start to (start + length) exclusively at dim.
static ElementsAttr tensorSlice(ElementsAttr tensor, uint64_t dim,
                                uint64_t start, uint64_t length) {
  auto shape = llvm::to_vector<4>(tensor.getType().getShape());
  if (length == shape[dim]) {
    // No need to slice.
    return tensor;
  }
  auto outputShape = shape;
  outputShape[dim] = length;
  auto outputType =
      RankedTensorType::get(outputShape, getElementTypeOrSelf(tensor));
  llvm::SmallVector<Attribute, 4> newContents;
  newContents.reserve(outputType.getNumElements());
  auto valuesBegin = tensor.getValues<Attribute>().begin();
  int64_t step =
      std::accumulate(shape.rbegin(), shape.rbegin() + shape.size() - dim,
                      /*init=*/1, /*op=*/std::multiplies<int64_t>());
  int64_t num = length * step / shape[dim];
  for (int64_t offset = step / shape[dim] * start,
               numElements = tensor.getType().getNumElements();
       offset < numElements; offset += step) {
    newContents.append(valuesBegin + offset, valuesBegin + offset + num);
  }
  return DenseElementsAttr::get(outputType, newContents);
}

OpFoldResult TensorSliceOp::fold(ArrayRef<Attribute> operands) {
  if (llvm::count(operands, nullptr) == 0) {
    // Fully constant arguments so we can perform the slice here.
    auto tensor = operands[0].cast<ElementsAttr>();
    int64_t rank = source().getType().cast<ShapedType>().getRank();
    // start = operands[1:1+rank), and length = operands[1+rank:].
    auto start = llvm::to_vector<4>(llvm::map_range(
        operands.drop_front(1).drop_back(rank), [](Attribute value) {
          return value.cast<IntegerAttr>().getValue().getZExtValue();
        }));
    auto length = llvm::to_vector<4>(
        llvm::map_range(operands.drop_front(1 + rank), [](Attribute value) {
          return value.cast<IntegerAttr>().getValue().getZExtValue();
        }));
    for (int64_t dim = 0; dim < rank; ++dim) {
      tensor = tensorSlice(tensor, dim, start[dim], length[dim]);
    }
    return tensor;
  }
  return {};
}

static ElementsAttr tensorUpdate(ElementsAttr update, ElementsAttr target,
                                 ArrayRef<Attribute> startIndicesAttrs) {
  auto updateType = update.getType().cast<ShapedType>();
  auto targetType = target.getType().cast<ShapedType>();
  // If either target or update has zero element, then no update happens.
  if (updateType.getNumElements() == 0 || targetType.getNumElements() == 0) {
    return target;
  }

  int64_t rank = targetType.getRank();
  // If target is scalar, update is also scalar and is the new content.
  if (rank == 0) {
    return update;
  }

  auto startIndex = llvm::to_vector<4>(
      llvm::map_range(startIndicesAttrs, [](Attribute value) {
        return value.cast<IntegerAttr>().getValue().getZExtValue();
      }));
  auto targetValues = llvm::to_vector<4>(target.getValues<Attribute>());
  // target indices start from startIndicesAttrs and update indices start from
  // all zeros.
  llvm::SmallVector<uint64_t, 4> targetIndex(startIndex);
  llvm::SmallVector<uint64_t, 4> updateIndex(rank, 0);
  int64_t numElements = updateType.getNumElements();
  while (numElements--) {
    targetValues[getFlattenedIndex(targetType, targetIndex)] =
        update.getValues<Attribute>()[updateIndex];
    // Increment the index at last dim.
    ++updateIndex.back();
    ++targetIndex.back();
    // If the index in dim j exceeds dim size, reset dim j and
    // increment dim (j-1).
    for (int64_t j = rank - 1;
         j >= 0 && updateIndex[j] >= updateType.getDimSize(j); --j) {
      updateIndex[j] = 0;
      targetIndex[j] = startIndex[j];
      if (j - 1 >= 0) {
        ++updateIndex[j - 1];
        ++targetIndex[j - 1];
      }
    }
  }
  return DenseElementsAttr::get(targetType, targetValues);
}

OpFoldResult TensorUpdateOp::fold(ArrayRef<Attribute> operands) {
  auto targetIndex = getODSOperandIndexAndLength(0).first;
  auto startIndices = getODSOperandIndexAndLength(2);
  auto updateIndex = getODSOperandIndexAndLength(3).first;
  auto indices = operands.slice(startIndices.first, startIndices.second);
  bool allIndicesConstant = llvm::count(indices, nullptr) == 0;
  if (operands[updateIndex] && operands[targetIndex] && allIndicesConstant) {
    // Fully constant arguments so we can perform the update here.
    return tensorUpdate(operands[updateIndex].cast<ElementsAttr>(),
                        operands[targetIndex].cast<ElementsAttr>(), indices);
  } else {
    // Replace the entire tensor when the sizes match.
    auto updateType = update().getType().cast<ShapedType>();
    auto targetType = target().getType().cast<ShapedType>();
    if (updateType.hasStaticShape() && targetType.hasStaticShape() &&
        updateType == targetType) {
      return update();
    }
  }
  return {};
}

namespace {

// When the target tensor is a result of a tensor.cast operation, the op needs
// to be updated to use the source of the cast as the target tensor.
struct FoldTensorUpdateOpWithCasts : public OpRewritePattern<TensorUpdateOp> {
  using OpRewritePattern<TensorUpdateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorUpdateOp updateOp,
                                PatternRewriter &rewriter) const override {
    auto targetCastOp = updateOp.target().getDefiningOp<tensor::CastOp>();
    auto updateCastOp = updateOp.update().getDefiningOp<tensor::CastOp>();
    if (!targetCastOp && !updateCastOp) return failure();
    auto target = (targetCastOp ? targetCastOp.source() : updateOp.target());
    auto update = (updateCastOp ? updateCastOp.source() : updateOp.update());
    auto newOp = rewriter.create<TensorUpdateOp>(
        updateOp.getLoc(), target.getType(), target,
        refreshDimsOnTypeChange(updateOp, updateOp.target().getType(),
                                target.getType(), updateOp.target_dims(),
                                rewriter),
        updateOp.start_indices(), update,
        refreshDimsOnTypeChange(updateOp, updateOp.update().getType(),
                                update.getType(), updateOp.update_dims(),
                                rewriter),
        updateOp.tied_operandsAttr());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        updateOp, updateOp.getResult().getType(), newOp.getResult());
    return success();
  }
};

}  // namespace

void TensorUpdateOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<FoldTensorUpdateOpWithCasts>(context);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
