#pragma once

#include "fixture.cuh"

// Honestly, we don't need fixture for this benchmark.

BENCHMARK_DEFINE_F(GpuFixture, BM_Morton)(bm::State& st) {
  const auto num_blocks = st.range(0);
  const auto block_size =
      DetermineBlockSizeAndDisplay(gpu::k_ComputeMortonCode, st);

  auto u_points_out = AllocateDevice<unsigned int>(kN);

  for (auto _ : st) {
    CudaEventTimer timer(st, true);

    gpu::k_ComputeMortonCode<<<num_blocks, block_size>>>(
        u_points, u_points_out, kN, kMin, kRange);
  }

  Free(u_points_out);
}

BENCHMARK_REGISTER_F(GpuFixture, BM_Morton)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 10)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);
