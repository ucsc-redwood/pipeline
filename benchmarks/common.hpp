
#pragma once

#include <algorithm>
#include <numeric>

#include "config.hpp"
#include "glm/glm.hpp"
#include "kernels/all.hpp"
#include "types/brt.hpp"

struct BenchmarkData {
  ~BenchmarkData() {
    delete[] morton_code;
    delete[] edge_count;
    delete[] count_prefix_sum;
    delete[] oct_nodes;
  }

  unsigned int* morton_code;

  RadixTreeData tree;

  int* edge_count;
  int* count_prefix_sum;
  OctNode* oct_nodes;
};

[[maybe_unused]]
static unsigned int* MakeSortedMortonReal(const int n) {
  auto data = new glm::vec4[n];
  k_InitRandomVec4Determinastic(data, n, kMin, kRange, 114514);
  auto morton_code = new unsigned int[n];
  k_ComputeMortonCode(data, morton_code, n, kMin, kRange);
  k_SortKeysInplace(morton_code, n);
  delete[] data;
  return morton_code;
}

[[nodiscard]] static unsigned int* MakeSortedMortonFake(const int n) {
  auto morton_code = new unsigned int[n];
  std::iota(morton_code, morton_code + n, 1);
  return morton_code;
}

static void MakeRadixTreeFake(unsigned int** morton_code, RadixTreeData& tree) {
  *morton_code = MakeSortedMortonFake(kN);

  const auto n_unique = static_cast<int>(0.98 * kN);

  tree.n_nodes = n_unique - 1;
  tree.prefixN = new uint8_t[tree.n_nodes];
  tree.hasLeafLeft = new bool[tree.n_nodes];
  tree.hasLeafRight = new bool[tree.n_nodes];
  tree.leftChild = new int[tree.n_nodes];
  tree.parent = new int[tree.n_nodes];

  k_BuildRadixTree(tree.n_nodes,
                   *morton_code,
                   tree.prefixN,
                   tree.hasLeafLeft,
                   tree.hasLeafRight,
                   tree.leftChild,
                   tree.parent);
}

// static void MakeRadixTreeAndPrefixSumFake(unsigned int** morton_code,
//                                           RadixTreeData& tree,
//                                           int** edge_count,
//                                           int** count_prefix_sum) {
//   MakeRadixTreeFake(morton_code, tree);

//   *edge_count = new int[tree.n_nodes];
//   k_EdgeCount(tree.prefixN, tree.parent, *edge_count, tree.n_nodes);

//   *count_prefix_sum = new int[tree.n_nodes + 1];
//   k_PartialSum(*edge_count, 0, tree.n_nodes, *count_prefix_sum);
//   *count_prefix_sum[0] = 0;
// }

[[nodiscard]] static BenchmarkData MakeRadixTreeAndPrefixSumFake() {
  BenchmarkData data;

  data.morton_code = MakeSortedMortonFake(kN);
  MakeRadixTreeFake(&data.morton_code, data.tree);

  data.tree.n_nodes = kN - 1;
  data.tree.prefixN = new uint8_t[data.tree.n_nodes];
  data.tree.hasLeafLeft = new bool[data.tree.n_nodes];
  data.tree.hasLeafRight = new bool[data.tree.n_nodes];
  data.tree.leftChild = new int[data.tree.n_nodes];
  data.tree.parent = new int[data.tree.n_nodes];

  data.edge_count = new int[data.tree.n_nodes];
  k_EdgeCount(
      data.tree.prefixN, data.tree.parent, data.edge_count, data.tree.n_nodes);

  data.count_prefix_sum = new int[data.tree.n_nodes + 1];
  k_PartialSum(data.edge_count, 0, data.tree.n_nodes, data.count_prefix_sum);
  data.count_prefix_sum[0] = 0;

  return data;
}
