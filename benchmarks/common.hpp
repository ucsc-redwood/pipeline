
#pragma once

#include <numeric>

#include "config.hpp"
#include "kernels/all.hpp"
#include "types/brt.hpp"

[[maybe_unused]] static unsigned int* MakeSortedMortonReal(const int n) {
  auto data = new glm::vec4[n];
  k_InitRandomVec4(data, n, kMin, kRange, 114514);
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
