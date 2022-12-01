// RUN: iree-opt %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<()[s0] -> (s0 * 4)>
#map3 = affine_map<()[s0] -> (s0 * 64)>


transform.structured.canonicalized_sequence failures(suppress) {
^bb1(%variant_op: !pdl.operation):
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op
  %reduction = transform.structured.match ops{["linalg.generic"]} in %variant_op

  // Step 1. First level of tiling + fusion parallelizes to blocks.
  // ===========================================================================
  %foreach_thread_grid, %grid_reduction =
    transform.iree.tile_to_foreach_thread_and_workgroup_count_region %reduction tile_sizes [1]
      ( mapping = [#gpu.block<x>] )
  transform.structured.fuse_into_containing_op %fill into %foreach_thread_grid

  // Step 2. Split the reduction to get meatier parallelism.
  // ===========================================================================
  %block_more_parallel_fill_op_2, %block_more_parallel_op_2, %block_combiner_op_2 = 
    transform.structured.tile_reduction_using_scf %grid_reduction { tile_sizes = [0, 2048] }
  %_1:2 =
    transform.structured.tile_to_foreach_thread_op %block_more_parallel_op_2 num_threads [1, 512] 
    ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )

  // Step 3. Second level of tiling parallelizes to threads.
  // ===========================================================================
  // 1st op is [parallel, parallel], map it to threadIdx.x by 4.
  %_2:2 =
    transform.structured.tile_to_foreach_thread_op %block_more_parallel_fill_op_2 tile_sizes [1, 4] 
    ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )
  // 2nd op is [parallel, reduction] of 1x128, map the 1-dim to threadIdx.y to
  // trigger mapping of the reduction to threadIdx.x via predication via `if (x==0)`.
  %_3:2 =
    transform.structured.tile_to_foreach_thread_op %block_combiner_op_2 tile_sizes [1] 
    ( mapping = [#gpu.thread<y>] )

  // Step 4. Rank-reduce and vectorize.
  // ===========================================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op
  %func_2 = transform.iree.apply_patterns %func { rank_reducing }
  %func_3 = transform.structured.vectorize %func_2

  // Step 5. Bufferize.
  // ===========================================================================
  %variant_op_2 = transform.iree.bufferize { target_gpu } %variant_op

  // Step 6. Post-bufferization mapping to blocks and threads.
  // ===========================================================================
  %func_4 = transform.structured.match ops{["func.func"]} in %variant_op_2
  %func_5 = transform.iree.foreach_thread_to_workgroup %func_4
  %func_6 = transform.iree.map_nested_foreach_thread_to_gpu_threads %func_5
      { workgroup_size = [512, 1, 1] }

  // Step 7. Post-bufferization vector distribution with rank-reduction.
  // ===========================================================================
  %func_7 = transform.iree.apply_patterns %func_6 { rank_reducing }
  %if_op = transform.structured.match ops{["scf.if"]} in %variant_op_2
  %warp = transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 }
  //transform.iree.vector.warp_distribute %func_7

  //%func_8 = transform.structured.match ops{["func.func"]} in %variant_op
  transform.structured.replace %func_7 {
        func.func @reduce_dispatch_0_generic_64x40960() {
          %c1_i32 = arith.constant 1 : i32
          %c2_i32 = arith.constant 2 : i32
          %c4_i32 = arith.constant 4 : i32
          %c8_i32 = arith.constant 8 : i32
          %c16_i32 = arith.constant 16 : i32
          %c32_i32 = arith.constant 32 : i32
          %c32 = arith.constant 32 : index
          %cst = arith.constant dense<-0.000000e+00> : vector<1xf32>
          %cst_0 = arith.constant dense<0.000000e+00> : vector<4xf32>
          %c40960 = arith.constant 40960 : index
          %cst_1 = arith.constant 0.000000e+00 : f32
          %c2048 = arith.constant 2048 : index
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<64x40960xf32>
          memref.assume_alignment %0, 64 : memref<64x40960xf32>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : memref<64xf32>
          memref.assume_alignment %1, 64 : memref<64xf32>
          %workgroup_id_x = hal.interface.workgroup.id[0] : index
          %subview = memref.subview %0[%workgroup_id_x, 0] [1, 40960] [1, 1] : memref<64x40960xf32> to memref<40960xf32, strided<[1], offset: ?>>
          %expand_shape = memref.expand_shape %subview [[0, 1]] : memref<40960xf32, strided<[1], offset: ?>> into memref<1x40960xf32, strided<[40960, 1], offset: ?>>
          %subview_2 = memref.subview %1[%workgroup_id_x] [1] [1] : memref<64xf32> to memref<f32, strided<[], offset: ?>>
          %expand_shape_3 = memref.expand_shape %subview_2 [] : memref<f32, strided<[], offset: ?>> into memref<1xf32, strided<[1], offset: ?>>
          vector.transfer_write %cst, %expand_shape_3[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>>
          %alloc = memref.alloc() {alignment = 128 : i64} : memref<1x2048xf32, 3>
          %2 = gpu.thread_id  x
          %3 = gpu.thread_id  y
          %4 = affine.apply #map2()[%2]
          %subview_4 = memref.subview %alloc[%3, %4] [1, 4] [1, 1] : memref<1x2048xf32, 3> to memref<4xf32, strided<[1], offset: ?>, 3>
          //vector.transfer_write %cst_0, %subview_4[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, strided<[1], offset: ?>, 3>
          gpu.barrier
          %v2 = scf.for %arg0 = %c0 to %c40960 step %c2048 iter_args(%v = %cst_0) -> (vector<4xf32>) {
            %subview_5 = memref.subview %expand_shape[0, %arg0] [1, 2048] [1, 1] : memref<1x40960xf32, strided<[40960, 1], offset: ?>> to memref<2048xf32, strided<[1], offset: ?>>
            %expand_shape_6 = memref.expand_shape %subview_5 [[0, 1]] : memref<2048xf32, strided<[1], offset: ?>> into memref<1x2048xf32, strided<[2048, 1], offset: ?>>
            %6 = vector.transfer_read %expand_shape_6[%3, %4], %cst_1 {in_bounds = [true]} : memref<1x2048xf32, strided<[2048, 1], offset: ?>>, vector<4xf32>
            //%7 = vector.transfer_read %alloc[%3, %4], %cst_1 {in_bounds = [true]} : memref<1x2048xf32, 3>, vector<4xf32>
            %8 = arith.addf %6, %v : vector<4xf32>
            //vector.transfer_write %8, %subview_4[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, strided<[1], offset: ?>, 3>
            scf.yield %8 : vector<4xf32>
          }
          vector.transfer_write %v2, %subview_4[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, strided<[1], offset: ?>, 3>
          gpu.barrier
          %5 = arith.cmpi ult, %2, %c32 : index
          scf.if %5 {
            %subview_5 = memref.subview %expand_shape_3[%3] [1] [1] : memref<1xf32, strided<[1], offset: ?>> to memref<f32, strided<[], offset: ?>>
            %6 = vector.transfer_read %subview_5[], %cst_1 : memref<f32, strided<[], offset: ?>>, vector<f32>
            %7 = affine.apply #map3()[%2]
            %8 = vector.transfer_read %alloc[%3, %7], %cst_1 {in_bounds = [true]} : memref<1x2048xf32, 3>, vector<64xf32>
            %9 = vector.extractelement %6[] : vector<f32>
            %10 = vector.reduction <add>, %8 : vector<64xf32> into f32
            %shuffleResult, %valid = gpu.shuffle  xor %10, %c1_i32, %c32_i32 : f32
            %11 = arith.addf %10, %shuffleResult : f32
            %shuffleResult_6, %valid_7 = gpu.shuffle  xor %11, %c2_i32, %c32_i32 : f32
            %12 = arith.addf %11, %shuffleResult_6 : f32
            %shuffleResult_8, %valid_9 = gpu.shuffle  xor %12, %c4_i32, %c32_i32 : f32
            %13 = arith.addf %12, %shuffleResult_8 : f32
            %shuffleResult_10, %valid_11 = gpu.shuffle  xor %13, %c8_i32, %c32_i32 : f32
            %14 = arith.addf %13, %shuffleResult_10 : f32
            %shuffleResult_12, %valid_13 = gpu.shuffle  xor %14, %c16_i32, %c32_i32 : f32
            %15 = arith.addf %14, %shuffleResult_12 : f32
            %16 = arith.addf %15, %9 : f32
            %17 = vector.broadcast %16 : f32 to vector<f32>
            %18 = arith.cmpi eq, %2, %c0 : index
            scf.if %18 {
              vector.transfer_write %17, %subview_5[] : vector<f32>, memref<f32, strided<[], offset: ?>>
            }
          }
          gpu.barrier
          memref.dealloc %alloc : memref<1x2048xf32, 3>
          return
        }
  }

}
