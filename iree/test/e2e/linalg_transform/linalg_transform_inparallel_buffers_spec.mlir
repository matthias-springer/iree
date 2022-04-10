// RUN: iree-opt %s 

pdl.pattern @pdl_matmul_target : benefit(1) {
  %args = operands
  %results = types
  %0 = operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
  // TODO: we don't want this, but it is the required terminator for pdl.pattern
  rewrite %0 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  %0 = match @pdl_matmul_target 
  print { name = "BEGIN"}

  %tiling_1_result:2 = tile_to_iree_linalg_ext_tile_op %0 {sizes = [2]}
  %inp_1 = rewrite_iree_linalg_ext_tile_to_in_parallel %tiling_1_result#1
  print { name = "AFTER inparallel"}

  iree_bufferize
  print { name = "AFTER bufferize"}

  iree_linalg_ext_inparallel_to_hal
  print { name = "AFTER InParallel to HAL"}
}
