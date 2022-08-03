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

// -----

func.func @main(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant {name = "constant.61"} dense<[[0.706495285, -0.567672312, 0.483717591, 0.522725761, 0.7563259], [-0.0899272263, -0.283501834, -0.350822538, -0.351515919, -0.337136656], [-0.451804549, 0.372324884, -0.620518147, 0.235451385, 0.851095855]]> : tensor<3x5xf32>
  %cst_1 = arith.constant {name = "constant.73"} dense<[[-0.0118641369, -3.785000e-02, 0.489048243, 0.321015775, -0.702280283], [-0.280262798, -0.724645615, -0.00332254497, 0.392334729, 0.619746447], [-0.113318317, -0.180415511, -0.146743968, 0.250408649, -0.442881733], [0.115600757, 0.703136146, -0.00812680274, -0.225454301, -0.0835619792], [-0.136745885, -6.298570e-01, 0.43629986, -0.689790308, 0.230725273]]> : tensor<5x5xf32>
  %cst_2 = arith.constant {name = "constant.85"} dense<[[-0.136191264, -0.0401721969, 0.38497138, -5.850760e-01, 0.370910525], [-0.391011149, 0.0266356133, 0.309115469, -0.205079094, -0.559861302], [0.497760415, 0.689488232, 0.0759292394, -0.33134672, -0.237128958], [-0.53243047, 0.476418108, -0.371978909, 0.283265263, 0.63842845], [0.101761498, -0.218626946, 0.475128263, 0.042601984, 0.0988005772]]> : tensor<5x5xf32>
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<1x5xf32>
  %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<1x5x3x1xf32>
  %2 = tensor.collapse_shape %0 [[0, 1]] : tensor<1x5xf32> into tensor<5xf32>
  %3 = linalg.init_tensor [5, 5] : tensor<5x5xf32>
  %4 = tensor.collapse_shape %1 [[0, 1], [2, 3]] : tensor<1x5x3x1xf32> into tensor<5x3xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%3 : tensor<5x5xf32>) -> tensor<5x5xf32>
  %6 = linalg.matmul {name = "dot.62"} ins(%4, %cst_0 : tensor<5x3xf32>, tensor<3x5xf32>) outs(%5 : tensor<5x5xf32>) -> tensor<5x5xf32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %6 : tensor<5xf32>, tensor<5x5xf32>) outs(%3 : tensor<5x5xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
    %16 = arith.maxf %arg3, %cst : f32
    %17 = arith.cmpf ogt, %arg2, %cst : f32
    %18 = arith.select %17, %cst, %16 : f32
    linalg.yield %18 : f32
  } -> tensor<5x5xf32>
  %8 = linalg.fill ins(%cst : f32) outs(%3 : tensor<5x5xf32>) -> tensor<5x5xf32>
  %9 = linalg.matmul {name = "dot.74"} ins(%7, %cst_1 : tensor<5x5xf32>, tensor<5x5xf32>) outs(%8 : tensor<5x5xf32>) -> tensor<5x5xf32>
  %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %9 : tensor<5xf32>, tensor<5x5xf32>) outs(%3 : tensor<5x5xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
    %16 = arith.maxf %arg3, %cst : f32
    %17 = arith.cmpf ogt, %arg2, %cst : f32
    %18 = arith.select %17, %cst, %16 : f32
    linalg.yield %18 : f32
  } -> tensor<5x5xf32>
  %11 = linalg.fill ins(%cst : f32) outs(%3 : tensor<5x5xf32>) -> tensor<5x5xf32>
  %12 = linalg.matmul {name = "dot.86"} ins(%10, %cst_2 : tensor<5x5xf32>, tensor<5x5xf32>) outs(%11 : tensor<5x5xf32>) -> tensor<5x5xf32>
  %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %12 : tensor<5xf32>, tensor<5x5xf32>) outs(%3 : tensor<5x5xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
    %16 = arith.cmpf ogt, %arg2, %cst : f32
    %17 = arith.maxf %arg3, %cst : f32
    %18 = arith.select %16, %cst, %17 : f32
    linalg.yield %18 : f32
  } -> tensor<5x5xf32>
  %14 = tensor.expand_shape %13 [[0, 1], [2]] : tensor<5x5xf32> into tensor<5x1x5xf32>
  %15 = hal.tensor.export %14 : tensor<5x1x5xf32> -> !hal.buffer_view
  return %15 : !hal.buffer_view
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %dispatch_op = transform.iree.make_dispatch_regions %arg1
  }
}
