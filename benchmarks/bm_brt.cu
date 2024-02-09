#include <benchmark/benchmark.h>

#include <glm/glm.hpp>

#include "config.hpp"
#include "cuda_bench_helper.cuh"
#include "kernels/init.hpp"

namespace bm = benchmark;

static void BM_Unique(bm::State& st) {
  const auto n_blocks = st.range(0);

  auto d_sort = AllocateDevice<unsigned int>(kN);

  gpu::k_InitAscendingSync(d_sort, kN);

  const auto num_unique = kN;

  for (auto _ : st) {
    cuda_event_timer timer(st, true);
  }
}

BENCHMARK(BM_Unique)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 10)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();

