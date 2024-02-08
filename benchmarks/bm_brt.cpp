#include <benchmark/benchmark.h>
#include <omp.h>

#include <algorithm>
#include <glm/glm.hpp>
#include <memory>

#include "common.hpp"
#include "config.hpp"
#include "data_loader.hpp"
#include "kernels/radix_tree.hpp"
#include "kernels/unique.hpp"
#include "types/brt.hpp"

namespace bm = benchmark;

static void BM_StdUnique(bm::State& st) {
  auto data = LoadSortedMortonCodes();

  for (auto _ : st) {
    [[maybe_unused]] auto last = std::unique(data, data + kN);
  }

  delete[] data;
}

static void BM_Unique(bm::State& st) {
  auto data = LoadSortedMortonCodes();

  for (auto _ : st) {
    [[maybe_unused]] auto last = k_CountUnique(data, kN);
  }

  delete[] data;
}

BENCHMARK(BM_StdUnique)->Unit(bm::kMillisecond);
BENCHMARK(BM_Unique)->Unit(bm::kMillisecond);

// -----------------------------------------------------

static void BM_RadixTree(bm::State& st) {
  const auto n_threads = st.range(0);

  auto data = LoadSortedMortonCodes();

  const auto last = std::unique(data, data + kN);
  const auto n_unique = std::distance(data, last);

  const auto tree = std::make_unique<RadixTree>(data, n_unique, kMin, kMax);

  omp_set_num_threads(n_threads);

  for (auto _ : st) {
    k_BuildRadixTree(tree.get());
  }

  delete[] data;
}

BENCHMARK(BM_RadixTree)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
