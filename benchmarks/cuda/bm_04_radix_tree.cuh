#pragma once

#include "fixture.cuh"

BENCHMARK_DEFINE_F(GpuFixture, BM_BuildRadixTree)(bm::State& st) {
  const auto num_blocks = st.range(0);
  const auto block_size =
      DetermineBlockSizeAndDisplay(gpu::k_BuildRadixTree, st);

  RadixTreeData out_tree;
  out_tree.n_nodes = num_unique - 1;
  out_tree.prefixN = AllocateManaged<uint8_t>(out_tree.n_nodes);
  out_tree.hasLeafLeft = AllocateManaged<bool>(out_tree.n_nodes);
  out_tree.hasLeafRight = AllocateManaged<bool>(out_tree.n_nodes);
  out_tree.leftChild = AllocateManaged<int>(out_tree.n_nodes);
  out_tree.parent = AllocateManaged<int>(out_tree.n_nodes);

  for (auto _ : st) {
    CudaEventTimer timer(st, true);

    gpu::k_BuildRadixTree<<<num_blocks, block_size>>>(num_unique,
                                                      u_sort,
                                                      out_tree.prefixN,
                                                      out_tree.hasLeafLeft,
                                                      out_tree.hasLeafRight,
                                                      out_tree.leftChild,
                                                      out_tree.parent);
  }

  Free(out_tree.prefixN);
  Free(out_tree.hasLeafLeft);
  Free(out_tree.hasLeafRight);
  Free(out_tree.leftChild);
  Free(out_tree.parent);
}

BENCHMARK_REGISTER_F(GpuFixture, BM_BuildRadixTree)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 10)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);
