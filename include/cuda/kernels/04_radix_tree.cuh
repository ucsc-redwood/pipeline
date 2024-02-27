#pragma once

#include <cstdint>

namespace gpu {

[[deprecated]] __global__ void k_BuildRadixTree(int n /* n_pts */,
                                                const unsigned int* codes,
                                                uint8_t* prefix_n,
                                                bool* has_leaf_left,
                                                bool* has_leaf_right,
                                                int* left_child,
                                                int* parent);

// We changed to use number of unique instead of number of nodes,
// in side the kernel we will subtract 1 to get the number of nodes
// The reason is we want to chain the kernel calls, and we don't want to do any
// subraction in the host code.
__global__ void k_BuildRadixTree_Deps(const int* n_unique /* n_unique */,
                                      const unsigned int* codes,
                                      uint8_t* prefix_n,
                                      bool* has_leaf_left,
                                      bool* has_leaf_right,
                                      int* left_child,
                                      int* parent);
}  // namespace gpu