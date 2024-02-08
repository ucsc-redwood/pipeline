#include <benchmark/benchmark.h>
#include <omp.h>

#include <algorithm>
#include <glm/glm.hpp>
#include <memory>

#include "common.hpp"
#include "config.hpp"
#include "kernels/radix_tree.hpp"
#include "kernels/unique.hpp"
#include "types/brt.hpp"

namespace bm = benchmark;

static void BM_StdUnique(bm::State& st) {
  auto morton_code = MakeSortedMortonFake(kN);

  for (auto _ : st) {
    auto last = std::unique(morton_code, morton_code + kN);
    bm::DoNotOptimize(last);
  }

  delete[] morton_code;
}

static void BM_Unique(bm::State& st) {
  auto morton_code = MakeSortedMortonFake(kN);

  for (auto _ : st) {
    auto last = k_CountUnique(morton_code, kN);
    bm::DoNotOptimize(last);
  }

  delete[] morton_code;
}

BENCHMARK(BM_StdUnique)->Unit(bm::kMillisecond);
BENCHMARK(BM_Unique)->Unit(bm::kMillisecond);

// -----------------------------------------------------

static void BM_RadixTree(bm::State& st) {
  const auto n_threads = st.range(0);

  auto morton_code = MakeSortedMortonFake(kN);

  // const auto last = std::unique(morton_code, morton_code + kN);
  // const auto n_unique = std::distance(morton_code, last);

  const auto n_unique = static_cast<int>(0.98 * kN);

  const auto tree =
      std::make_unique<RadixTree>(morton_code, n_unique, kMin, kMax);

  omp_set_num_threads(n_threads);

  for (auto _ : st) {
    k_BuildRadixTree(tree.get());
  }

  delete[] morton_code;
}

BENCHMARK(BM_RadixTree)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
