// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_TRANSFORMEXTENSIONS_FLOWEXTENSIONS
#define IREE_COMPILER_DIALECT_FLOW_TRANSFORMEXTENSIONS_FLOWEXTENSIONS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformEffects.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def ForeachThreadToFlowDispatchWorkgroupsOp : Op<Transform_Dialect, "iree.foreach_thread_to_flow",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformOpInterface,
     TransformEachOpTrait]> {
  let description = [{
    Rewrite an scf.foreach_thread to Flow::DispatchWorkgroups.


    Return modes:
    =============
    This operation ignores non-scf::ForeachThread ops and drops them in the
    return.

    If any rewrite fails, the transform definitely fails.

    If all the operations referred to by the `target` PDLOperation generalize
    properly, the transform succeeds. Otherwise the transform silently fails.

    The return handle points to only the subset of successfully produced
    equivalent flow::DispatchWorkgroups operations, which can be empty.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::scf::ForeachThreadOp target,
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];
}

def WrapInDispatchRegionOp : Op<
    Transform_Dialect, "iree.wrap_in_dispatch_region",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Wrap the `target` op in a new `dispatch.region` op. All uses of target op
    are replaces with the results of the newly generated `dispach.region` op.

    #### Return modes

    This transform consumes the `target` handle and produces the `transformed`
    handle (i.e., the `dispatch.region` op).
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];
}

def FlowDispatchRegionToFlowDispatchWorkgroupsOp
    : Op<Transform_Dialect, "iree.region_to_workgroups",
        [FunctionalStyleTransformOpTrait,
         TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Rewrite `flow.dispatch.region` ops to `flow.dispatch.workgroups` ops.
    The former ones allow arbitrary SSA values that are in scope to be captured
    inside the body region, whereas the latter ones are isolated from above.

    This transforms clones all ops in the `target` payload op body into a new
    `workgroups` op. It analyzes which SSA values from outside the region are
    used inside the region and passes them explicitly as operands. Inside the
    region, they are accessed via block arguments. Tensor operands are wrapped
    in `!flow.dispatch.tensor` types inside the `workgroups` op region.

    #### Return modes

    This operation consumes (reads and frees) the `target` handle. It produces
    (allocates and writes) the result handle. All `target` payload ops must be
    `flow.dispatch.region` ops, otherwise, the transform fails.
   }];

  let arguments = (ins Arg<PDL_Operation, "",
                           [TransformMappingRead,
                            TransformMappingFree]>:$target);
  let results = (outs Res<PDL_Operation, "",
                          [TransformMappingAlloc,
                           TransformMappingWrite]>:$transformed);
  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::iree_compiler::IREE::Flow::DispatchRegionOp target,
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];
}

def MoveIntoDispatchRegionOp : Op<
    Transform_Dialect, "iree.move_into_dispatch_region",
    [TransformOpInterface]> {
  let description = [{
    Clone the `target` op into the given dispatch region op.

    If the target op dominates the dispatch region op, it is transformed only
    if there is at least one consumer inside the region. Otherwise, the
    transform fails. (Can be extended in the future.) Uses of the original
    target outside of the dispatch region remain unmodified. If there are no
    such uses, the original target folds away.

    If the target op post-dominates the dispatch region, it is always cloned
    into the region. A new result is added to the dispatch region and all
    original uses of the target op are replaced with the newly added result.
    The original target op is removed. This transform fails if not all uses of
    original target dominate (or are equal to) the dispatch region op.

    #### Return modes

    This transform consumes both the `target` handle and the `dispatch_region`
    handle. It produces a new handle to the extended dispatch region.
  }];

  let arguments = (ins Arg<PDL_Operation, "",
                           [TransformMappingRead,
                            TransformMappingFree]>:$target,
                       Arg<PDL_Operation, "",
                           [TransformMappingRead,
                            TransformMappingFree]>:$dispatch_region);
  let results = (outs Res<PDL_Operation, "",
                           [TransformMappingAlloc,
                            TransformMappingWrite]>:$transformed);
  let assemblyFormat = "$target `into` $dispatch_region attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];
}

def MakeDispatchRegionsOp : Op<
    Transform_Dialect, "iree.make_dispatch_regions",
    [TransformOpInterface]> {
  let description = [{ Make dispatch regions }];
  let arguments = (ins Arg<PDL_Operation, "",
                           [TransformMappingRead]>:$target);
  let results = (outs Res<PDL_Operation, "",
                           [TransformMappingAlloc,
                            TransformMappingWrite]>:$dispatch_region);
  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // IREE_COMPILER_DIALECT_FLOW_TRANSFORMEXTENSIONS_FLOWEXTENSIONS
