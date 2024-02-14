#pragma once

#include <cstdint>

namespace gpu {

__global__ void k_BuildRadixTree(int n /* n_pts */,
                                 const unsigned int* codes,
                                 uint8_t* prefix_n,
                                 bool* has_leaf_left,
                                 bool* has_leaf_right,
                                 int* left_child,
                                 int* parent);
}