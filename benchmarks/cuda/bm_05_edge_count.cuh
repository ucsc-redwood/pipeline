#pragma once

#include "fixture.cuh"

BENCHMARK_DEFINE_F(GpuFixture, BM_EdgeCount)(bm::State& st) {
  const auto num_blocks = st.range(0);
  const auto block_size =
      DetermineBlockSizeAndDisplay(gpu::k_EdgeCount, st);

  auto u_edge_count_out = AllocateManaged<int>(tree.n_nodes);

  for (auto _ : st) {
    CudaEventTimer timer(st, true);

    gpu::k_EdgeCount<<<num_blocks, block_size>>>(
        tree.prefixN, tree.parent, u_edge_count_out, tree.n_nodes);
  }

  Free(u_edge_count_out);
}

BENCHMARK_REGISTER_F(GpuFixture, BM_EdgeCount)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 10)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);
