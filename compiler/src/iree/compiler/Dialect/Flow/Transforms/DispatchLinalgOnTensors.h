// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_DISPATCHLINALGONTENSORS_H_
#define IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_DISPATCHLINALGONTENSORS_H_

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class Operation;

namespace iree_compiler {
namespace IREE {
namespace Flow {

/// A heuristic that decides which ops should be cloned and fused into a
/// dispatch region.
///
/// Note: This function returns `false` for ops that should be tiled and fused
/// into a dispatch region.
bool isClonableIntoDispatchOp(Operation *op);

/// Reorders the operations in `ops` such that they could be inlined into the
/// dispatch region in that order to satisfy dependencies.
llvm::SmallVector<Operation *> orderOperations(llvm::ArrayRef<Operation *> ops);

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_DISPATCHLINALGONTENSORS_H_
