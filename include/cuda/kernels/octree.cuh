#pragma once

#include "types/oct.cuh"

namespace gpu {

__global__ void k_MakeOctNodes(gpu::OctNode* oct_nodes,
                               const int* node_offsets,    // prefix sum
                               const int* rt_node_counts,  // edge count
                               const unsigned int* codes,
                               const uint8_t* rt_prefixN,
                               const int* rt_parents,
                               float min_coord,
                               float range,
                               int N  // number of brt nodes
);

__global__ void k_LinkLeafNodes(gpu::OctNode* nodes,
                                const int* node_offsets,
                                const int* rt_node_counts,
                                const unsigned int* codes,
                                const bool* rt_hasLeafLeft,
                                const bool* rt_hasLeafRight,
                                const uint8_t* rt_prefixN,
                                const int* rt_parents,
                                const int* rt_leftChild,
                                int N);

}  // namespace gpu
