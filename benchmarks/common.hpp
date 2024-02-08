
#pragma once

#include <algorithm>
#include <cassert>
#include <memory>

#include "config.hpp"
#include "data_loader.hpp"
#include "kernels/all.hpp"
//#include "types/brt.hpp"

// Wrapper for easy to create morton codes
[[nodiscard]] static unsigned int* LoadSortedMortonCodes() {
  size_t size;
  auto data = loadFromBinaryFile<unsigned int>(
      "data/bm_sorted_mortons_u32_10m.bin", size);
  assert(size == kN);

  return data;
}

[[nodiscard]] static std::unique_ptr<RadixTree> CreateRadixTree() {
  auto data = LoadSortedMortonCodes();

  const auto last = std::unique(data, data + kN);
  const auto n_unique = std::distance(data, last);

  // return std::make_unique<RadixTree>(data, n_unique, kMin, kMax);

  auto tree = std::make_unique<RadixTree>(data, n_unique, kMin, kMax);
  k_BuildRadixTree(tree.get());

  delete[] data;
  return tree;
}
