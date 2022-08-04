// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule -canonicalize -cse -split-input-file | FileCheck %s

// CHECK-LABEL: func @foo(
//       CHECK:   %[[argA:.*]]: tensor<?x?xf32>, %[[argB:.*]]: tensor<5x10xf32>, %[[argC:.*]]: tensor<10x11xf32>
func.func @foo(%argA: tensor<?x?xf32>, %argB: tensor<5x10xf32>, %argC: tensor<10x11xf32>) -> (tensor<?x?xf32>, tensor<5x11xf32>) {
  //  CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  //  CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  //  CHECK-DAG: %[[dim_argA_0:.*]] = tensor.dim %[[argA]], %[[c0]]
  //  CHECK-DAG: %[[dim_argA_1:.*]] = tensor.dim %[[argA]], %[[c1]]
  //      CHECK: %[[r0:.*]] = flow.dispatch.workgroups(%[[argA]], %[[dim_argA_0]], %[[dim_argA_1]]) : (tensor<?x?xf32>{%[[dim_argA_0]], %[[dim_argA_1]]}, index, index) -> %[[argA]]{%[[dim_argA_0]], %[[dim_argA_1]]} =
  // CHECK-NEXT: (%[[arg1:.*]]: !flow.dispatch.tensor<readwrite:?x?xf32>, %[[arg2:.*]]: index, %[[arg3:.*]]: index) {
  //      CHECK:   %[[load:.*]] = flow.dispatch.tensor.load %[[arg1]], offsets = [0, 0], sizes = [%[[arg2]], %[[arg3]]], strides = [1, 1] : !flow.dispatch.tensor<readwrite:?x?xf32>{%[[arg2]], %[[arg3]]} -> tensor<?x?xf32>
  //      CHECK:   flow.dispatch.tensor.store %[[load]], %[[arg1]], offsets = [0, 0], sizes = [%[[arg2]], %[[arg3]]], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:?x?xf32>{%[[arg2]], %[[arg3]]}
  //      CHECK:   flow.return
  //      CHECK: }
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dimA0 = tensor.dim %argA, %c0 : tensor<?x?xf32>
  %dimA1 = tensor.dim %argA, %c1 : tensor<?x?xf32>
  %r0 = flow.dispatch.region -> (tensor<?x?xf32>{%dimA0, %dimA1}) {
    flow.return %argA : tensor<?x?xf32>
  }

  //      CHECK: %[[r1:.*]] = flow.dispatch.workgroups(%[[argB]], %[[argC]]) : (tensor<5x10xf32>, tensor<10x11xf32>) -> tensor<5x11xf32> =
  // CHECK-NEXT: (%[[arg3:.*]]: !flow.dispatch.tensor<readonly:5x10xf32>, %[[arg4:.*]]: !flow.dispatch.tensor<readonly:10x11xf32>, %[[arg5:.*]]: !flow.dispatch.tensor<writeonly:5x11xf32>)
  //  CHECK-DAG:   %[[loadB:.*]] = flow.dispatch.tensor.load %[[arg3]], offsets = [0, 0], sizes = [5, 10], strides = [1, 1] : !flow.dispatch.tensor<readonly:5x10xf32> -> tensor<5x10xf32>
  //  CHECK-DAG:   %[[loadC:.*]] = flow.dispatch.tensor.load %[[arg4]], offsets = [0, 0], sizes = [10, 11], strides = [1, 1] : !flow.dispatch.tensor<readonly:10x11xf32> -> tensor<10x11xf32>
  //      CHECK:   %[[init_tensor:.*]] = linalg.init_tensor [5, 11] : tensor<5x11xf32>
  //      CHECK:   %[[fill:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[init_tensor]] : tensor<5x11xf32>) -> tensor<5x11xf32>
  //      CHECK:   %[[matmul:.*]] = linalg.matmul ins(%[[loadB]], %[[loadC]] : tensor<5x10xf32>, tensor<10x11xf32>) outs(%[[fill]] : tensor<5x11xf32>) -> tensor<5x11xf32>
  //      CHECK:   flow.dispatch.tensor.store %[[matmul]], %[[arg5]], offsets = [0, 0], sizes = [5, 11], strides = [1, 1] : tensor<5x11xf32> -> !flow.dispatch.tensor<writeonly:5x11xf32>
  //      CHECK:   flow.return
  //      CHECK: }
  %r1 = flow.dispatch.region -> (tensor<5x11xf32>) {
    %zero = arith.constant 0.0 : f32
    %0 = linalg.init_tensor [5, 11] : tensor<5x11xf32>
    %1 = linalg.fill ins(%zero : f32) outs(%0 : tensor<5x11xf32>) -> tensor<5x11xf32>
    %2 = linalg.matmul ins(%argB, %argC : tensor<5x10xf32>, tensor<10x11xf32>)
        outs(%1 : tensor<5x11xf32>) -> tensor<5x11xf32>
    flow.return %2 : tensor<5x11xf32>
  }

  //      CHECK: return %[[r0]], %[[r1]]
  return %r0, %r1 : tensor<?x?xf32>, tensor<5x11xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.iree.match ops{["flow.dispatch.region"]} in %arg1
    transform.iree.region_to_workgroups %0
  }
}
