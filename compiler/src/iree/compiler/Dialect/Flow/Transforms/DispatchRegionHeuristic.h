// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_DISPATCHREGIONHEURISTIC_H_
#define IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_DISPATCHREGIONHEURISTIC_H_

namespace mlir {
class DominanceInfo;
class Operation;

namespace iree_compiler {
namespace IREE {
namespace Flow {

/// Returns true if an op has a root operation.
bool hasRootOpAttribute(Operation *op);

/// Removes root attribute. Asserts if root attribute is not present.
void removeRootOpAttribute(Operation *op);

/// Returns the number of the root. Asserts if the operation is not already set
/// as a root.
int64_t getRootNumber(Operation *op);

/// Returns true if an op is part of a fusion group.
bool hasFusionGroupsAttribute(Operation *op);

/// Returns the fusion groups for the given `op`.
SmallVector<int64_t, 1> getFusionGroups(Operation *op);

/// Returns true if the given `op` is in the `targetGroup` fusion group.
bool isInFusionGroup(Operation *op, unsigned targetGroup);

/// Removes the fusion groups attribute.
void removeFusionGroupsAttribute(Operation *op);

/// Determine fusion groups.
unsigned decideFusableLinalgOps(FunctionOpInterface funcOp,
                                DominanceInfo const &dominanceInfo,
                                bool aggressiveFusion);

/// A heuristic that decides which ops should be cloned and fused into a
/// dispatch region.
///
/// Note: This function returns `false` for ops that should be tiled and fused
/// into a dispatch region.
bool isClonableIntoDispatchOp(Operation *op);

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_DISPATCHREGIONHEURISTIC_H_
