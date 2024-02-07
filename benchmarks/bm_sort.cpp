#include <benchmark/benchmark.h>

#include <glm/glm.hpp>

#include "config.hpp"
#include "kernels/sort.hpp"

namespace bm = benchmark;

static void BM_SortUint32(bm::State& st) {
  auto data = new unsigned int[kN];

  for (auto _ : st) {
    k_SortKeysInplace(data, kN);
    bm::DoNotOptimize(data[0]);
  }

  delete[] data;
}

BENCHMARK(BM_SortUint32)->Unit(bm::kMillisecond);

BENCHMARK_MAIN();

