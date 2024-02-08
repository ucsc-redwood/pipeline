#include <benchmark/benchmark.h>
#include <omp.h>

#include "common.hpp"
#include "kernels/all.hpp"

namespace bm = benchmark;

// static void BM_MakeOctNodes(bm::State& st) {
//   const auto n_threads = st.range(0);

//   const auto u_input = new glm::vec4[kN];
//   auto u_sort = new unsigned int[kN];

//   constexpr auto seed = 114514;
//   k_InitRandomVec4Determinastic(u_input, kN, kMin, kRange, seed);

//   k_ComputeMortonCode(u_input, u_sort, kN, kMin, kRange);

//   k_SortKeysInplace(u_sort, kN);

//   const auto n_unique = k_CountUnique(u_sort, kN);

//   RadixTreeData tree;
//   tree.n_nodes = n_unique - 1;
//   tree.prefixN = new uint8_t[tree.n_nodes];
//   tree.hasLeafLeft = new bool[tree.n_nodes];
//   tree.hasLeafRight = new bool[tree.n_nodes];
//   tree.leftChild = new int[tree.n_nodes];
//   tree.parent = new int[tree.n_nodes];

//   auto edge_count = new int[tree.n_nodes];

//   k_EdgeCount(tree.prefixN, tree.parent, edge_count, tree.n_nodes);
//   auto count_prefix_sum = new int[tree.n_nodes + 1];
//   k_PartialSum(edge_count, 0, tree.n_nodes, count_prefix_sum);
//   count_prefix_sum[0] = 0;

//   const auto num_oct_nodes = count_prefix_sum[tree.n_nodes];

//   auto oct_nodes = new OctNode[num_oct_nodes];

//   st.counters["n_oct_nodes"] = num_oct_nodes;

//   omp_set_num_threads(n_threads);

//   for (auto _ : st) {
//     const auto root_level = tree.prefixN[0] / 3;
//     const auto root_prefix = u_sort[0] >> (morton_bits - (3 * root_level));

//     morton32_to_xyz(&oct_nodes[0].cornor,
//                     root_prefix << (morton_bits - (3 * root_level)),
//                     kMin,
//                     kRange);
//     oct_nodes[0].cell_size = kRange;

//     k_MakeOctNodes(oct_nodes,
//                    count_prefix_sum,
//                    edge_count,
//                    u_sort,
//                    tree.prefixN,
//                    tree.parent,
//                    kMin,
//                    kRange,
//                    num_oct_nodes);
//   }

//   delete[] oct_nodes;
//   delete[] count_prefix_sum;
//   delete[] edge_count;
//   delete[] tree.parent;
//   delete[] tree.leftChild;
//   delete[] tree.hasLeafRight;
//   delete[] tree.hasLeafLeft;
//   delete[] tree.prefixN;
//   delete[] u_sort;
//   delete[] u_input;
// }

static void BM_MakeOctNodes(bm::State& st) {
  const auto n_threads = st.range(0);

  constexpr auto n = kN;
  constexpr auto min = 0.0f;
  constexpr auto max = 1024.0f;
  constexpr auto range = max - min;

  //   const auto u_input = new glm::vec4[n];
  //   auto u_sort = new unsigned int[n];

  //   constexpr auto seed = 114514;
  //   k_InitRandomVec4Determinastic(u_input, n, min, range, seed);

  //   k_ComputeMortonCode(u_input, u_sort, n, min, range);

  //   k_SortKeysInplace(u_sort, n);

  auto u_sort = MakeSortedMortonFake(n);

  const auto n_unique = k_CountUnique(u_sort, n);

  RadixTreeData tree;
  tree.n_nodes = n_unique - 1;
  tree.prefixN = new uint8_t[tree.n_nodes];
  tree.hasLeafLeft = new bool[tree.n_nodes];
  tree.hasLeafRight = new bool[tree.n_nodes];
  tree.leftChild = new int[tree.n_nodes];
  tree.parent = new int[tree.n_nodes];

  k_BuildRadixTree(tree.n_nodes,
                   u_sort,
                   tree.prefixN,
                   tree.hasLeafLeft,
                   tree.hasLeafRight,
                   tree.leftChild,
                   tree.parent);

  auto u_edge_count = new int[tree.n_nodes];

  k_EdgeCount(tree.prefixN, tree.parent, u_edge_count, n_unique);

  auto u_count_prefix_sum = new int[n_unique];

  [[maybe_unused]] auto _ =
      k_PartialSum(u_edge_count, 0, n_unique, u_count_prefix_sum);
  u_count_prefix_sum[0] = 0;

  const auto num_oct_nodes = u_count_prefix_sum[tree.n_nodes];

  auto u_oct_nodes = new OctNode[num_oct_nodes];

  const auto root_level = tree.prefixN[0] / 3;
  const auto root_prefix = u_sort[0] >> (morton_bits - (3 * root_level));

  morton32_to_xyz(&u_oct_nodes[0].cornor,
                  root_prefix << (morton_bits - (3 * root_level)),
                  min,
                  range);
  u_oct_nodes[0].cell_size = range;

  omp_set_num_threads(n_threads);

  for (auto _ : st) {
    k_MakeOctNodes(u_oct_nodes,
                   u_count_prefix_sum,
                   u_edge_count,
                   u_sort,
                   tree.prefixN,
                   tree.parent,
                   min,
                   range,
                   num_oct_nodes);
  }

  delete[] u_oct_nodes;
  delete[] u_count_prefix_sum;
  delete[] u_edge_count;
  //   delete[] tree.parent;
  //   delete[] tree.leftChild;
  //   delete[] tree.hasLeafRight;
  //   delete[] tree.hasLeafLeft;
  //   delete[] tree.prefixN;
  delete[] u_sort;
  //   delete[] u_input;
}

static void BM_LinkOctreeNodes(bm::State& st) {
  const auto n_threads = st.range(0);

  constexpr auto n = kN;
  constexpr auto min = 0.0f;
  constexpr auto max = 1024.0f;
  constexpr auto range = max - min;

  //   const auto u_input = new glm::vec4[n];
  //   auto u_sort = new unsigned int[n];

  //   constexpr auto seed = 114514;
  //   k_InitRandomVec4Determinastic(u_input, n, min, range, seed);

  //   k_ComputeMortonCode(u_input, u_sort, n, min, range);

  //   k_SortKeysInplace(u_sort, n);

  auto u_sort = MakeSortedMortonFake(n);

  const auto n_unique = k_CountUnique(u_sort, n);

  RadixTreeData tree;
  tree.n_nodes = n_unique - 1;
  tree.prefixN = new uint8_t[tree.n_nodes];
  tree.hasLeafLeft = new bool[tree.n_nodes];
  tree.hasLeafRight = new bool[tree.n_nodes];
  tree.leftChild = new int[tree.n_nodes];
  tree.parent = new int[tree.n_nodes];

  k_BuildRadixTree(tree.n_nodes,
                   u_sort,
                   tree.prefixN,
                   tree.hasLeafLeft,
                   tree.hasLeafRight,
                   tree.leftChild,
                   tree.parent);

  auto u_edge_count = new int[tree.n_nodes];

  k_EdgeCount(tree.prefixN, tree.parent, u_edge_count, n_unique);

  auto u_count_prefix_sum = new int[n_unique];

  [[maybe_unused]] auto _ =
      k_PartialSum(u_edge_count, 0, n_unique, u_count_prefix_sum);
  u_count_prefix_sum[0] = 0;

  const auto num_oct_nodes = u_count_prefix_sum[tree.n_nodes];

  auto u_oct_nodes = new OctNode[num_oct_nodes];

  const auto root_level = tree.prefixN[0] / 3;
  const auto root_prefix = u_sort[0] >> (morton_bits - (3 * root_level));

  morton32_to_xyz(&u_oct_nodes[0].cornor,
                  root_prefix << (morton_bits - (3 * root_level)),
                  min,
                  range);
  u_oct_nodes[0].cell_size = range;

  k_MakeOctNodes(u_oct_nodes,
                 u_count_prefix_sum,
                 u_edge_count,
                 u_sort,
                 tree.prefixN,
                 tree.parent,
                 min,
                 range,
                 num_oct_nodes);

  omp_set_num_threads(n_threads);

  for (auto _ : st) {
    k_LinkLeafNodes(u_oct_nodes,
                    u_count_prefix_sum,
                    u_edge_count,
                    u_sort,
                    tree.hasLeafLeft,
                    tree.hasLeafRight,
                    tree.prefixN,
                    tree.parent,
                    tree.leftChild,
                    num_oct_nodes);
  }

  delete[] u_oct_nodes;
  delete[] u_count_prefix_sum;
  delete[] u_edge_count;
  //   delete[] tree.parent;
  //   delete[] tree.leftChild;
  //   delete[] tree.hasLeafRight;
  //   delete[] tree.hasLeafLeft;
  //   delete[] tree.prefixN;
  delete[] u_sort;
  //   delete[] u_input;
}

BENCHMARK(BM_MakeOctNodes)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_LinkOctreeNodes)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
