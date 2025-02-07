// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_IREE_TRANSFORMS_PATTERNS_H_
#define IREE_COMPILER_DIALECT_IREE_TRANSFORMS_PATTERNS_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

// Populates |patterns| with some risky/IREE-specific canonicalization patterns.
// Some of these apply to other dialects (such as std/builtin) and could be
// upstreamed after some more exhaustive investigation.
void populateCommonPatterns(MLIRContext *context,
                            OwningRewritePatternList &patterns);

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_IREE_TRANSFORMS_PATTERNS_H_
