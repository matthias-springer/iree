// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule -split-input-file | FileCheck %s

// CHECK-LABEL: func @single_op(
//  CHECK-SAME:   %[[arg0:.*]]: tensor<?x?xf32>, %[[s1:.*]]: index, %[[s2:.*]]: index
func.func @single_op(%arg0: tensor<?x?xf32>, %s1: index, %s2: index) -> tensor<?x?xf32> {
  // CHECK: %[[region:.*]] = flow.dispatch.region -> (tensor<?x?xf32>{%[[s1]], %[[s2]]}) {
  // CHECK:   %[[slice:.*]] = tensor.extract_slice %[[arg0]]
  // CHECK:   flow.return %[[slice]]
  // CHECK: }
  // CHECK: return %[[region]]
  %0 = tensor.extract_slice %arg0 [0, 10] [%s1, %s2] [1, 1]
      : tensor<?x?xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.iree.match ops{["tensor.extract_slice"]} in %arg1
    transform.iree.wrap_in_dispatch_region %0
  }
}

// -----

// CHECK-LABEL: func @fuse_producer(
//  CHECK-SAME:   %[[arg0:.*]]: tensor<?x?xf32>, %[[arg1:.*]]: tensor<?x?xf32>, %[[s1:.*]]: index, %[[s2:.*]]: index
func.func @fuse_producer(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %s1: index, %s2: index) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[dim0:.*]] = tensor.dim %[[arg1]], %[[c0]]
  // CHECK-DAG: %[[dim1:.*]] = tensor.dim %[[arg1]], %[[c1]]
  // CHECK: %[[region:.*]] = flow.dispatch.region -> (tensor<?x?xf32>{%[[dim0]], %[[dim1]]}) {
  // CHECK:   %[[slice:.*]] = tensor.extract_slice %[[arg0]]
  // CHECK:   %[[insert:.*]] = tensor.insert_slice %[[slice]] into %[[arg1]]
  // CHECK:   flow.return %[[insert]]
  // CHECK: }
  // CHECK: return %[[region]]
  %0 = tensor.extract_slice %arg0 [0, 10] [%s1, %s2] [1, 1]
      : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = tensor.insert_slice %0 into %arg1 [5, 16] [%s1, %s2] [1, 1]
      : tensor<?x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.iree.match ops{["tensor.insert_slice"]} in %arg1
    %dispatch_op = transform.iree.wrap_in_dispatch_region %0
    %1 = transform.iree.match ops{["tensor.extract_slice"]} in %arg1
    transform.iree.move_into_dispatch_region %1 into %dispatch_op
  }
}

// -----

// CHECK-LABEL: func @fuse_consumer(
//  CHECK-SAME:   %[[arg0:.*]]: tensor<?x?xf32>, %[[arg1:.*]]: tensor<?x?xf32>, %[[s1:.*]]: index, %[[s2:.*]]: index
func.func @fuse_consumer(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %s1: index, %s2: index) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[dim0:.*]] = tensor.dim %[[arg1]], %[[c0]]
  // CHECK-DAG: %[[dim1:.*]] = tensor.dim %[[arg1]], %[[c1]]
  // CHECK: %[[region:.*]]:2 = flow.dispatch.region -> (tensor<?x?xf32>{%[[s1]], %[[s2]]}, tensor<?x?xf32>{%[[dim0]], %[[dim1]]}) {
  // CHECK:   %[[slice:.*]] = tensor.extract_slice %[[arg0]]
  // CHECK:   %[[insert:.*]] = tensor.insert_slice %[[slice]] into %[[arg1]]
  // CHECK:   flow.return %[[slice]], %[[insert]]
  // CHECK: }
  // CHECK: return %[[region]]#1
  %0 = tensor.extract_slice %arg0 [0, 10] [%s1, %s2] [1, 1]
      : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = tensor.insert_slice %0 into %arg1 [5, 16] [%s1, %s2] [1, 1]
      : tensor<?x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.iree.match ops{["tensor.extract_slice"]} in %arg1
    %dispatch_op = transform.iree.wrap_in_dispatch_region %0
    %1 = transform.iree.match ops{["tensor.insert_slice"]} in %arg1
    transform.iree.move_into_dispatch_region %1 into %dispatch_op
  }
}

// -----

// CHECK-LABEL: func @fuse_consumer2(
//  CHECK-SAME:   %[[arg0:.*]]: tensor<?x?xf32>, %[[arg1:.*]]: tensor<?x?xf32>, %[[s1:.*]]: index, %[[s2:.*]]: index
func.func @fuse_consumer2(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %s1: index, %s2: index) -> tensor<?x?xf32> {
  // CHECK: %[[region:.*]]:2 = flow.dispatch.region -> (tensor<?x?xf32>{%[[s1]], %[[s2]]}, tensor<?x?xf32>{%[[s1]], %[[s2]]}) {
  // CHECK:   %[[slice:.*]] = tensor.extract_slice %[[arg0]]
  // CHECK:   %[[insert:.*]] = tensor.insert_slice %[[arg1]] into %[[slice]]
  // CHECK:   flow.return %[[slice]], %[[insert]]
  // CHECK: }
  // CHECK: return %[[region]]#1
  %0 = tensor.extract_slice %arg0 [0, 10] [%s1, %s2] [1, 1]
      : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = tensor.insert_slice %arg1 into %0 [5, 16] [%s1, %s2] [1, 1]
      : tensor<?x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.iree.match ops{["tensor.extract_slice"]} in %arg1
    %dispatch_op = transform.iree.wrap_in_dispatch_region %0
    %1 = transform.iree.match ops{["tensor.insert_slice"]} in %arg1
    transform.iree.move_into_dispatch_region %1 into %dispatch_op
  }
}
