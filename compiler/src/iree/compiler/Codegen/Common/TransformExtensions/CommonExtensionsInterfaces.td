// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_COMMONEXTENSIONSINTERFACES
#define IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_COMMONEXTENSIONSINTERFACES

include "mlir/IR/OpBase.td"

def MatchConstraintOpInterface : OpInterface<"MatchConstraintOpInterface"> {
  let description = [{
    Ops that implement this interface can be used as constraints in the body of
    a iree.match op.
  }];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return `true` if the given op satisfies this constraint.
      }],
      /*retType=*/"bool",
      /*methodName=*/"satisfied",
      /*args=*/(ins "Operation *":$op)
    >,
  ];
  
  let verify = [{
    return ::mlir::iree_compiler::IREE::transform_dialect::detail
        ::verifyMatchConstraintOpInterface($_op);
  }];
}

#endif // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_COMMONEXTENSIONSINTERFACES
