#include <benchmark/benchmark.h>
#include <omp.h>

#include <algorithm>
#include <glm/glm.hpp>

#include "common.hpp"
#include "config.hpp"
#include "kernels/all.hpp"

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

  const auto n_unique = static_cast<int>(0.98 * kN);

  auto n_nodes = n_unique - 1;
  auto prefixN = new uint8_t[n_nodes];
  auto hasLeafLeft = new bool[n_nodes];
  auto hasLeafRight = new bool[n_nodes];
  auto leftChild = new int[n_nodes];
  auto parent = new int[n_nodes];

  omp_set_num_threads(n_threads);

  for (auto _ : st) {
    k_BuildRadixTree(n_nodes,
                     morton_code,
                     prefixN,
                     hasLeafLeft,
                     hasLeafRight,
                     leftChild,
                     parent);
  }

  delete[] parent;
  delete[] leftChild;
  delete[] hasLeafRight;
  delete[] hasLeafLeft;
  delete[] prefixN;
  delete[] morton_code;
}

static void BM_EdgeCount(bm::State& st) {
  const auto n_threads = st.range(0);

  // prepare a radix tree
  unsigned int* morton_code = nullptr;
  RadixTreeData tree;
  MakeRadixTreeFake(morton_code, tree);

  auto edge_count = new int[tree.n_nodes];

  omp_set_num_threads(n_threads);

  for (auto _ : st) {
    k_EdgeCount(tree.prefixN, tree.parent, edge_count, tree.n_nodes);
  }

  delete[] edge_count;
  delete[] morton_code;
}

static void BM_PrefixSum(bm::State& st) {
  std::vector<int> edge_count(kN);
  std::vector<int> count_prefix_sum(kN);

  std::iota(edge_count.begin(), edge_count.end(), 0);

  for (auto _ : st) {
    k_PartialSum(edge_count.data(), 0, kN, count_prefix_sum.data());
  }
}

static void BM_PrefixSum_Std(bm::State& st) {
  std::vector<int> edge_count(kN);
  std::vector<int> count_prefix_sum(kN);

  std::iota(edge_count.begin(), edge_count.end(), 0);

  for (auto _ : st) {
    std::partial_sum(
        edge_count.begin(), edge_count.end(), count_prefix_sum.begin());
  }
}

BENCHMARK(BM_RadixTree)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_EdgeCount)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_PrefixSum)->Unit(bm::kMillisecond);

BENCHMARK(BM_PrefixSum_Std)->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
