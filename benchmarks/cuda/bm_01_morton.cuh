#pragma once

#include "fixture.cuh"

BENCHMARK_DEFINE_F(GpuFixture, BM_Morton32)(bm::State& st) {
  const auto num_blocks = st.range(0);
  const auto block_size =
      DetermineBlockSizeAndDisplay(gpu::k_ComputeMortonCode, st);

  // const auto num_threads = optimal_block_size["morton"];
  // st.counters["threads"] = num_threads;

  for (auto _ : st) {
    CudaEventTimer timer(st, true);

    // 'u_points' -> (process) -> 'u_points_out'
    gpu::k_ComputeMortonCode<<<num_blocks, 768>>>(
        u_points, u_points_out, kN, kMin, kRange);
  }
}

BENCHMARK_REGISTER_F(GpuFixture, BM_Morton32)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 10)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);
