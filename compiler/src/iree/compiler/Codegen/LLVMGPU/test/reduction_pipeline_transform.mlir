// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))" --iree-codegen-llvmgpu-enable-transform-dialect-jit %s | FileCheck %s

hal.executable @group_reduction {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @group_reduction ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_reduction() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<8x64xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x64xf32>> -> tensor<8x64xf32>
      %3 = tensor.empty() : tensor<8xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<8x64xf32>) outs(%4 : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %out : f32
        linalg.yield %6 : f32
      } -> tensor<8xf32>
      flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_reduction
//     CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//     CHECK-DAG:   %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index
//     CHECK-DAG:   %[[SHMEM_ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x32xf32, 3>
//     CHECK-DAG:   %[[TIDX:.]] = gpu.thread_id  x

// Fusion occurred, no barrier before the loop
//     CHECK-NOT: gpu.barrier
//     CHECK:   vector.transfer_read {{.*}} vector<f32>
// Local per-thread scf.for-based reduction.
//         CHECK: scf.for
//         CHECK:   vector.transfer_read {{.*}} vector<2xf32>
//         CHECK:   vector.reduction <add>{{.*}} : vector<2xf32> into f32
//         CHECK:   vector.broadcast {{.*}} : f32 to vector<f32>
// No barrier within the loop
//     CHECK-NOT:   gpu.barrier
//         CHECK:   scf.yield {{.*}} : vector<f32>

// Distributed reduction: everyone loads then 5 xor + addf expected.
//         CHECK: %[[TIDY:.]] = gpu.thread_id  y
//     CHECK-DAG: vector.transfer_read %{{.*}}[%{{.*}}]
//     CHECK-DAG: vector.transfer_read %{{.*}}[%[[TIDY]], %[[TIDX]]]
// CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

//         CHECK:   %[[RES:.*]] = arith.addf %{{.*}}
//         CHECK:   %[[RES_VEC:.*]] = vector.broadcast %[[RES]] : f32 to vector<f32>
//         CHECK:   %[[CONDXIS0:.*]] = arith.cmpi eq, %[[TIDX]], %[[C0]] : index
//         CHECK:   scf.if %[[CONDXIS0]]
//         CHECK:     vector.transfer_write %[[RES_VEC]]
//         CHECK:   gpu.barrier
//         CHECK:   memref.dealloc %[[SHMEM_ALLOC]]

// -----

hal.executable @group_reduction_elementwise {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @group_reduction_elementwise ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_reduction_elementwise() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<8x64xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x64xf32>> -> tensor<8x64xf32>
      %3 = tensor.empty() : tensor<8xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<8x64xf32>) outs(%4 : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %7 = arith.addf %in, %out : f32
        linalg.yield %7 : f32
      } -> tensor<8xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%5 : tensor<8xf32>) outs(%3 : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %7 = math.sqrt %in : f32
        linalg.yield %7 : f32
      } -> tensor<8xf32>
      flow.dispatch.tensor.store %6, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_reduction_elementwise
//     CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//     CHECK-DAG:   %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index
//     CHECK-DAG:   %[[SHMEM_ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x32xf32, 3>
//     CHECK-DAG:   %[[TIDX:.]] = gpu.thread_id  x

// Fusion occurred, no barrier before the loop
//     CHECK-NOT: gpu.barrier
//     CHECK:   vector.transfer_read {{.*}} vector<f32>
// Local per-thread scf.for-based reduction.
//         CHECK: scf.for
//         CHECK:   vector.transfer_read {{.*}} vector<2xf32>
//         CHECK:   vector.reduction <add>{{.*}} : vector<2xf32> into f32
//         CHECK:   vector.broadcast {{.*}} : f32 to vector<f32>
// No barrier within the loop
//     CHECK-NOT:   gpu.barrier
//         CHECK:   scf.yield {{.*}} : vector<f32>

// Distributed reduction: everyone loads then 5 xor + addf expected.
//         CHECK: %[[TIDY:.]] = gpu.thread_id  y
//     CHECK-DAG: vector.transfer_read %{{.*}}[%{{.*}}]
//     CHECK-DAG: vector.transfer_read %{{.*}}[%[[TIDY]], %[[TIDX]]]
// CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

//         CHECK:   %[[PARTIAL:.*]] = arith.addf %{{.*}}
//         CHECK:   vector.broadcast %[[PARTIAL]] : f32 to vector<f32>
//         CHECK:   math.sqrt 
//         CHECK:   %[[RES_VEC:.*]] = vector.broadcast %{{.*}}: f32 to vector<f32>
//         CHECK:   %[[CONDXIS0:.*]] = arith.cmpi eq, %[[TIDX]], %[[C0]] : index
//         CHECK:   scf.if %[[CONDXIS0]]
//         CHECK:     vector.transfer_write %[[RES_VEC]]
//         CHECK:   gpu.barrier
//         CHECK:   memref.dealloc %[[SHMEM_ALLOC]]

// -----

hal.executable @group_elementwise_reduction {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @group_elementwise_reduction ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_elementwise_reduction() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<8x64xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x64xf32>> -> tensor<8x64xf32>
      %3 = tensor.empty() : tensor<8xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<8x64xf32>) outs(%4 : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %in : f32
        %7 = arith.addf %6, %6 : f32
        %8 = arith.addf %7, %out : f32
        linalg.yield %8 : f32
      } -> tensor<8xf32>
      flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_elementwise_reduction
//     CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//     CHECK-DAG:   %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index
//     CHECK-DAG:   %[[SHMEM_ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x32xf32, 3>
//     CHECK-DAG:   %[[TIDX:.]] = gpu.thread_id  x

// Fusion occurred, no barrier before the loop
//     CHECK-NOT: gpu.barrier
//     CHECK:   vector.transfer_read {{.*}} vector<f32>
// Local per-thread scf.for-based reduction.
//         CHECK: scf.for
//         CHECK:   vector.transfer_read {{.*}} vector<2xf32>
//         CHECK:   arith.addf{{.*}} : vector<2xf32>
//         CHECK:   arith.addf{{.*}} : vector<2xf32>
//         CHECK:   vector.reduction <add>{{.*}} : vector<2xf32> into f32
//         CHECK:   vector.broadcast {{.*}} : f32 to vector<f32>
// No barrier within the loop
//     CHECK-NOT:   gpu.barrier
//         CHECK:   scf.yield {{.*}} : vector<f32>

// Distributed reduction: everyone loads then 5 xor + addf expected.
//         CHECK: %[[TIDY:.]] = gpu.thread_id  y
//     CHECK-DAG: vector.transfer_read %{{.*}}[%{{.*}}]
//     CHECK-DAG: vector.transfer_read %{{.*}}[%[[TIDY]], %[[TIDX]]]
// CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

//         CHECK:   %[[RES:.*]] = arith.addf %{{.*}}
//         CHECK:   %[[RES_VEC:.*]] = vector.broadcast %[[RES]] : f32 to vector<f32>
//         CHECK:   %[[CONDXIS0:.*]] = arith.cmpi eq, %[[TIDX]], %[[C0]] : index
//         CHECK:   scf.if %[[CONDXIS0]]
//         CHECK:     vector.transfer_write %[[RES_VEC]]
//         CHECK:   gpu.barrier
//         CHECK:   memref.dealloc %[[SHMEM_ALLOC]]

// -----

hal.executable @group_elementwise_reduction_elementwise {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @group_elementwise_reduction_elementwise ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_elementwise_reduction_elementwise() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<8x64xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x64xf32>> -> tensor<8x64xf32>
      %3 = tensor.empty() : tensor<8xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<8x64xf32>) outs(%4 : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %7 = arith.addf %in, %in : f32
        %8 = arith.addf %7, %7 : f32
        %9 = arith.addf %8, %out : f32
        linalg.yield %9 : f32
      } -> tensor<8xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%5 : tensor<8xf32>) outs(%3 : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %7 = math.sqrt %in : f32
        linalg.yield %7 : f32
      } -> tensor<8xf32>
      flow.dispatch.tensor.store %6, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_elementwise_reduction_elementwise
//     CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//     CHECK-DAG:   %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index
//     CHECK-DAG:   %[[SHMEM_ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x32xf32, 3>
//     CHECK-DAG:   %[[TIDX:.]] = gpu.thread_id  x

// Fusion occurred, no barrier before the loop
//     CHECK-NOT: gpu.barrier
//     CHECK:   vector.transfer_read {{.*}} vector<f32>
// Local per-thread scf.for-based reduction.
//         CHECK: scf.for
//         CHECK:   vector.transfer_read {{.*}} vector<2xf32>
//         CHECK:   arith.addf{{.*}} : vector<2xf32>
//         CHECK:   arith.addf{{.*}} : vector<2xf32>
//         CHECK:   vector.reduction <add>{{.*}} : vector<2xf32> into f32
//         CHECK:   vector.broadcast {{.*}} : f32 to vector<f32>
// No barrier within the loop
//     CHECK-NOT:   gpu.barrier
//         CHECK:   scf.yield {{.*}} : vector<f32>

// Distributed reduction: everyone loads then 5 xor + addf expected.
//         CHECK: %[[TIDY:.]] = gpu.thread_id  y
//     CHECK-DAG: vector.transfer_read %{{.*}}[%{{.*}}]
//     CHECK-DAG: vector.transfer_read %{{.*}}[%[[TIDY]], %[[TIDX]]]
// CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

//         CHECK:   %[[PARTIAL:.*]] = arith.addf %{{.*}}
//         CHECK:   vector.broadcast %[[PARTIAL]] : f32 to vector<f32>
//         CHECK:   math.sqrt
//         CHECK:   %[[RES_VEC:.*]] = vector.broadcast %{{.*}}: f32 to vector<f32>
//         CHECK:   %[[CONDXIS0:.*]] = arith.cmpi eq, %[[TIDX]], %[[C0]] : index
//         CHECK:   scf.if %[[CONDXIS0]]
//         CHECK:     vector.transfer_write %[[RES_VEC]]
//         CHECK:   gpu.barrier
//         CHECK:   memref.dealloc %[[SHMEM_ALLOC]]

// -----

hal.executable @group_reduction_larger {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @group_reduction_larger ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_reduction_larger() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<33x256xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<33xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [33, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<33x256xf32>> -> tensor<33x256xf32>
      %3 = tensor.empty() : tensor<33xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<33xf32>) -> tensor<33xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<33x256xf32>) outs(%4 : tensor<33xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %out : f32
        linalg.yield %6 : f32
      } -> tensor<33xf32>
      flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [8], strides = [1] : tensor<33xf32> -> !flow.dispatch.tensor<writeonly:tensor<33xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_reduction_larger
//     CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//     CHECK-DAG:   %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index
//     CHECK-DAG:   %[[SHMEM_ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x64xf32, 3>
//     CHECK-DAG:   %[[TIDX:.]] = gpu.thread_id  x

// Fusion occurred, no barrier before the loop
//     CHECK-NOT: gpu.barrier
//     CHECK:   vector.transfer_read {{.*}} vector<f32>
// Local per-thread scf.for-based reduction.
//         CHECK: scf.for
//         CHECK:   vector.transfer_read
//         CHECK:   vector.reduction <add>{{.*}} : vector<4xf32> into f32
//         CHECK:   vector.broadcast {{.*}} : f32 to vector<f32>
// No barrier within the loop
//     CHECK-NOT:   gpu.barrier
//         CHECK:   scf.yield {{.*}} : vector<f32>

// Distributed reduction: everyone loads then 5 xor + addf expected.
//         CHECK: %[[TIDY:.]] = gpu.thread_id  y
//         CHECK: vector.transfer_read %{{.*}}[%{{.*}}]
//         CHECK: %[[IDX:.*]] = affine.apply{{.*}}%[[TIDX]]
//         CHECK: vector.transfer_read %{{.*}}[%[[TIDY]], %[[IDX]]]
// CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

//         CHECK:   %[[RES:.*]] = arith.addf %{{.*}}
//         CHECK:   %[[RES_VEC:.*]] = vector.broadcast %[[RES]] : f32 to vector<f32>
//         CHECK:   %[[CONDXIS0:.*]] = arith.cmpi eq, %[[TIDX]], %[[C0]] : index
//         CHECK:   scf.if %[[CONDXIS0]]
//         CHECK:     vector.transfer_write %[[RES_VEC]]
//         CHECK:   gpu.barrier
//         CHECK:   memref.dealloc %[[SHMEM_ALLOC]]
