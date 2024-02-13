#pragma once

#include <cstdint>

namespace gpu {

__global__ void k_BuildRadixTree_Kernel(const int n /* n_pts */,
                                        const unsigned int* codes,
                                        uint8_t* prefix_n,
                                        bool* has_leaf_left,
                                        bool* has_leaf_right,
                                        int* left_child,
                                        int* parent);

// User specified number of blocks
void Dispatch_BuildRadixTree_With(const int n /* n_pts */,
                                  const unsigned int* codes,
                                  uint8_t* prefix_n,
                                  bool* has_leaf_left,
                                  bool* has_leaf_right,
                                  int* left_child,
                                  int* parent,
                                  int logical_num_blocks /* gpu_thing */);

// // using CUDA's API to determine block size
// void Dispatch_BuildRadixTree_WithOptimizedBlock(const int n /* n_pts */,
//                                            const unsigned int* codes,
//                                            uint8_t* prefix_n,
//                                            bool* has_leaf_left,
//                                            bool* has_leaf_right,
//                                            int* left_child,
//                                            int* parent);

}  // namespace gpu