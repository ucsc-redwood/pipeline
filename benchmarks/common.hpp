
#pragma once

#include <algorithm>
// #include <memory>
#include <numeric>

#include "config.hpp"
#include "glm/glm.hpp"
#include "kernels/all.hpp"

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

// Wrapper for easy to create morton codes
static void MakeRadixTree(unsigned int* morton_keys) {
  morton_keys = MakeSortedMortonFake(kN);
  
  // auto tree = std::make_unique<RadixTree>(morton_keys, kN, kMin, kMax);
  // k_BuildRadixTree(tree.get());



}

// static unsigned int* MakeSortedMortonFake(const int n) {
//   auto morton_keys = new unsigned int[n];

// #pragma omp parallel for
//   for (auto i = 0; i < n; i++) {
//     morton_keys[i] = i;
//   }

//   return morton_keys;
// }

// #include <cassert>
// #include <memory>

// #include "config.hpp"
// #include "data_loader.hpp"
// #include "kernels/all.hpp"
// //#include "types/brt.hpp"

// [[nodiscard]] static std::unique_ptr<RadixTree> CreateRadixTree() {
//   auto data = LoadSortedMortonCodes();

//   const auto last = std::unique(data, data + kN);
//   const auto n_unique = std::distance(data, last);

//   // return std::make_unique<RadixTree>(data, n_unique, kMin, kMax);

//   auto tree = std::make_unique<RadixTree>(data, n_unique, kMin, kMax);
//   k_BuildRadixTree(tree.get());

//   delete[] data;
//   return tree;
// }
