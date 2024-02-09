#include <benchmark/benchmark.h>

#include <glm/glm.hpp>

#include "config.hpp"
#include "cuda/kernels/morton.cuh"
#include "cuda_bench_helper.cuh"
#include "kernels/init.hpp"

namespace bm = benchmark;

static void BM_Morton32(bm::State& st) {
  const auto n_blocks = st.range(0);
  const auto block_size =
      DetermineBlockSizeAndDisplay(gpu::k_ComputeMorton_Kernel, st);

  const auto d_xyz = AllocateDevice<glm::vec4>(kN);
  const auto d_morton = AllocateDevice<unsigned int>(kN);

  gpu::k_InitRandomVec4(d_xyz, kN, kMin, kRange, kRandomSeed);
  BENCH_CUDA_TRY(cudaDeviceSynchronize());

  for (auto _ : st) {
    cuda_event_timer timer(st, true);

    gpu::k_ComputeMorton_Kernel<<<n_blocks, block_size>>>(
        d_xyz, d_morton, kN, kMin, kRange);
  }

  Free(d_xyz);
  Free(d_morton);
}

BENCHMARK(BM_Morton32)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 10)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
