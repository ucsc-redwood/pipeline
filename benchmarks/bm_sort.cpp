#include <benchmark/benchmark.h>
#include <omp.h>

#include <glm/glm.hpp>

#include "config.hpp"
#include "kernels/init.hpp"
#include "kernels/sort.hpp"

namespace bm = benchmark;

static void BM_SortUint32(bm::State& st) {
  auto data = new unsigned int[kN];

  k_InitDescending(data, kN);

  for (auto _ : st) {
    k_SortKeysInplace(data, kN);
    bm::DoNotOptimize(data[0]);
  }

  delete[] data;
}

static void BM_SimpleRadixSort(bm::State& st) {
  const auto n_threads = st.range(0);
  auto data = new unsigned int[kN];

  k_InitDescending(data, kN);

  omp_set_num_threads(n_threads);

  for (auto _ : st) {
    k_SimpleRadixSort(data, kN);
    bm::DoNotOptimize(data[0]);
  }

  delete[] data;
}

static void BM_SimpleRadixSort_WithBuffer(bm::State& st) {
  const auto n_threads = st.range(0);
  auto data = new unsigned int[kN];
  auto data_alt = new unsigned int[kN];

  k_InitDescending(data, kN);

  omp_set_num_threads(n_threads);

  for (auto _ : st) {
    k_SimpleRadixSort(data, data_alt, kN);
    bm::DoNotOptimize(data[0]);
  }

  delete[] data;
  delete[] data_alt;
}

BENCHMARK(BM_SortUint32)->Unit(bm::kMillisecond);

BENCHMARK(BM_SimpleRadixSort)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_SimpleRadixSort_WithBuffer)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
