transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @pdl_matmul_target : benefit(1) {
    %args = operands
    %results = types
    %0 = operation (%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    pdl.apply_native_constraint "implementsTilingInterface"(%0 : !pdl.operation)
    pdl.apply_native_constraint "hasNoConsumingLinalgPointwise"(%0 : !pdl.operation)
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @pdl_matmul_target in %arg1
    %tiling_1_result:2 = tile_to_foreach_thread_op %0 {num_threads = [13, 33], fuse_producers_greedily = true}
    transform.iree.foreach_thread_to_flow
  }
}
