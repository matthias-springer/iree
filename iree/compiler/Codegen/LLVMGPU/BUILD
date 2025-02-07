# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "LLVMGPU",
    srcs = [
        "ConvertToLLVM.cpp",
        "ConvertToNVVM.cpp",
        "ConvertToROCDL.cpp",
        "KernelConfig.cpp",
        "LLVMGPUDistributeSharedMemoryCopy.cpp",
        "LLVMGPULowerExecutableTarget.cpp",
        "LLVMGPUPipelining.cpp",
        "LLVMGPUTensorCoreVectorization.cpp",
        "LLVMGPUTileAndDistribute.cpp",
        "LLVMGPUUtils.cpp",
        "LLVMGPUVectorLowering.cpp",
        "LLVMGPUVectorization.cpp",
        "Passes.cpp",
    ],
    hdrs = [
        "ConvertToLLVM.h",
        "KernelConfig.h",
        "LLVMGPUUtils.h",
    ],
    deps = [
        "//iree/compiler/Codegen:PassHeaders",
        "//iree/compiler/Codegen/Common",
        "//iree/compiler/Codegen/Dialect:IREECodegenDialect",
        "//iree/compiler/Codegen/Transforms",
        "//iree/compiler/Codegen/Utils",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/Util/IR",
        "//llvm-external-projects/iree-dialects:IREELinalgExtDialect",
        "//llvm-external-projects/iree-dialects:IREELinalgExtTransforms",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ArithmeticToLLVM",
        "@llvm-project//mlir:ArithmeticTransforms",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToNVVMTransforms",
        "@llvm-project//mlir:GPUToROCDLTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToStandard",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorOps",
        "@llvm-project//mlir:VectorToGPU",
        "@llvm-project//mlir:VectorToLLVM",
        "@llvm-project//mlir:VectorToSCF",
    ],
)
