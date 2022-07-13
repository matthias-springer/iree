// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_COMMONEXTENSIONSOPS
#define IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_COMMONEXTENSIONSOPS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "CommonExtensionsInterfaces.td"

def ApplyPatternsOp : Op<Transform_Dialect, "iree.apply_patterns",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformEachOpTrait, 
     TransformOpInterface]> {
  let description = [{
    Greedily applies patterns as specified by its attributes.
    
    Must be applied to an op with trait IsolatedFromAbove since the 
    GreedyPatternRewriter asserts those.

    Returns the IsolatedFromAbove op whose content it has modified for better
    chaining APIs.

    The following additive attributes can be set, they add patterns in an 
    unspecified order:
      - canonicalization: adds all the canonicalization pattenrs of all 
      registered dialects and ops.
      - rank_reducing: adds patterns that results in rank-reducing behavior on
      subset-based operations.

    Return modes:
    =============
    This operation applies a number of patterns to rewrite vector IR into 
    distributed warp form. To apply these patterns, this operation must target 
    an operation that is isolated from above, otherwise the transform definitely
    fails.
  
    If the pattern application fails, or if the underlying listener fails to 
    capture op handles, the transformation definitely fails.

    Otherwise the transformation is successful and no result is returned.
  }];

  let arguments = (ins PDL_Operation:$target,
                       UnitAttr:$canonicalization,
                       UnitAttr:$rank_reducing);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
        ::mlir::transform::TransformState &state);
  }];
}

def IREEBufferizeOp : Op<Transform_Dialect, "iree.bufferize",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Call upstream comprehensive bufferize with extra IREE hooks.

    By default, CPU allocations are emitted. This behavior can be modified by 
    using the following attributes:
      - target_gpu: if set, GPU allocations are emitted.

    Return modes:
    =============
    This operation calls the upstream one-shot bufferization pass with extra
    registered patterns for IREE.
  
    The pass is ran on all the ModuleOp nested under the top-level op on which
    the transform dialect interpreter pass is applied.

    If any of the pass on any of the ModuleOp fails, the transformation 
    definitely fails. Otherwise the transformation succeeds.

    No handles are consumed or produced.
  }];

  let arguments = (ins UnitAttr:$target_gpu);
  let results = (outs);

  let assemblyFormat = "attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}

def MatchInterfaceEnum : I32EnumAttr<"MatchInterfaceEnum", "An interface to match",
    [
      I32EnumAttrCase<"LinalgOp", 0>,
      I32EnumAttrCase<"TilingInterface", 1>
    ]>{
  let cppNamespace = "::mlir::iree_compiler::IREE::transform_dialect";
}

// TODO: Should not implement TransformOpInterface and
// MemoryEffectsOpInterface, but only ops that implement these interfaces
// can be injected into the transform dialect.
def MatchTerminatorOp : Op<Transform_Dialect, "iree.match.terminator",
    [TransformOpInterface, MemoryEffectsOpInterface, Terminator]> {
  let description = "terminator iree.match region";
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = [{attr-dict}];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state) {
      return ::mlir::DiagnosedSilenceableFailure::definiteFailure();
    }

    void getEffects(
        SmallVectorImpl<::mlir::MemoryEffects::EffectInstance> &effects) {}
  }];
}

def MatchOp : Op<Transform_Dialect, "iree.match",
    [IsolatedFromAbove,
     MemoryEffectsOpInterface,
     NavigationTransformOpTrait,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    This op traverses the ops nested under `target` and returns the handles to
    all the operations that match the requirements. Only ops that satisfy all
    constraints in the region body are matched.

    All ops in the region body must implement the MatchConstraintOpInterface.
    The region may be empty. In that case, all ops within the target op are
    matched.

    This op fails if the target is not a handle to exactly one operation. 
    Otherwise it succeeds.
  
    This operation does not consume the target handle and produces new handles:
    it is a navigation op.
  }];

  let arguments = (ins PDL_Operation:$target);
  // TODO: veriadic results when needed.
  let results = (outs PDL_Operation:$results);
  let regions = (region AnyRegion:$constraints);

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    BlockArgument getMatchedOp() {
      return getBody()->getArguments().front();
    }

    Block *getBody() {
      if (getConstraints().getBlocks().empty())
        return nullptr;
      return &getConstraints().getBlocks().front();
    }
  }];
}

// TODO: Should not implement TransformOpInterface and
// MemoryEffectsOpInterface, but only ops that implement these interfaces
// can be injected into the transform dialect.
def MatchIsAOp : Op<Transform_Dialect, "iree.match.isa",
    [TransformOpInterface, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<MatchConstraintOpInterface>]> {
  let description = [{
    Constrains the name / interface of an op.

    The following variants are supported:
      - interface: an optional MatchInterfaceEnum specifying an enum
      representation for an interface to target.
      - ops: an optional StrArrayAttr specifying the concrete name of an op. 
      
    Note: either `ops` or `interface` must be specified.
  }];

  let arguments = (ins PDL_Operation:$target,
                       OptionalAttr<StrArrayAttr>:$match_op,
                       OptionalAttr<MatchInterfaceEnum>:$match_interface);
  let results = (outs);
  let assemblyFormat = [{
    $target `,`
    (`ops` `{` $match_op^ `}`)? 
    (`interface` `{` $match_interface^ `}`)? 
    attr-dict
  }];

  let hasVerifier = 1;
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state) {
      return ::mlir::DiagnosedSilenceableFailure::definiteFailure();
    }

    void getEffects(
        SmallVectorImpl<::mlir::MemoryEffects::EffectInstance> &effects) {}
  }];
}
#endif // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_COMMONEXTENSIONSOPS
