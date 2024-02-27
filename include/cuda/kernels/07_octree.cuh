#pragma once

#include "shared/types.h"

namespace gpu {

namespace v2 {

__global__ void k_MakeOctNodes_Deps(
    // --- new parameters
    int (*u_children)[8],
    glm::vec4* u_corner,
    float* u_cell_size,
    int* u_child_node_mask,
    // --- end new parameters
    const int* node_offsets,    // prefix sum
    const int* rt_node_counts,  // edge count
    const unsigned int* codes,
    const uint8_t* rt_prefixN,
    const int* rt_parents,
    const float min_coord,
    const float range,
    // const int N /* number of brt nodes */
    const int* u_num_unique);

__global__ void k_LinkLeafNodes_Deps(
    // --- new parameters
    int (*u_children)[8],
    glm::vec4* u_corner,
    float* u_cell_size,
    int* u_child_leaf_mask,
    // --- end new parameters
    const int* node_offsets,
    const int* rt_node_counts,
    const unsigned int* codes,
    const bool* rt_hasLeafLeft,
    const bool* rt_hasLeafRight,
    const uint8_t* rt_prefixN,
    const int* rt_parents,
    const int* rt_leftChild,
    const int* u_num_unique);

}  // namespace v2

[[deprecated]] __global__ void k_MakeOctNodes(
    OctNode* oct_nodes,
    const int* node_offsets,    // prefix sum
    const int* rt_node_counts,  // edge count
    const unsigned int* codes,
    const uint8_t* rt_prefixN,
    const int* rt_parents,
    float min_coord,
    float range,
    int N  // number of brt nodes ? really? Yanwen: Yes
);

[[deprecated]] __global__ void k_LinkLeafNodes(OctNode* nodes,
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