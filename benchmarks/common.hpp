
#pragma once

#include <algorithm>
#include <numeric>

#include "config.hpp"
#include "glm/glm.hpp"
#include "kernels/all.hpp"
#include "types/brt.hpp"

static unsigned int* MakeSortedMortonReal(const int n) {
  auto data = new glm::vec4[n];
  k_InitRandomVec4Determinastic(data, n, kMin, kRange, 114514);
  auto morton_keys = new unsigned int[n];
  k_ComputeMortonCode(data, morton_keys, n, kMin, kRange);
  k_SortKeysInplace(morton_keys, n);
  delete[] data;
  return morton_keys;
}

[[nodiscard]] static unsigned int* MakeSortedMortonFake(const int n) {
  auto morton_keys = new unsigned int[n];
  std::iota(morton_keys, morton_keys + n, 1);
  return morton_keys;
}

static void MakeRadixTreeFake(unsigned int* morton_keys, RadixTreeData& tree) {
  morton_keys = MakeSortedMortonFake(kN);

  const auto n_unique = static_cast<int>(0.98 * kN);

  tree.n_nodes = n_unique - 1;
  tree.prefixN = new uint8_t[tree.n_nodes];
  tree.hasLeafLeft = new bool[tree.n_nodes];
  tree.hasLeafRight = new bool[tree.n_nodes];
  tree.leftChild = new int[tree.n_nodes];
  tree.parent = new int[tree.n_nodes];

  k_BuildRadixTree(tree.n_nodes,
                   morton_keys,
                   tree.prefixN,
                   tree.hasLeafLeft,
                   tree.hasLeafRight,
                   tree.leftChild,
                   tree.parent);
}
