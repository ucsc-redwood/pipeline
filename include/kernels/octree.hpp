#pragma once

#include "types/morton.hpp"
#include "types/oct.hpp"

void k_MakeOctNodes(OctNode* oct_nodes,
                    const int* node_offsets,    // prefix sum
                    const int* rt_node_counts,  // edge count
                    const MortonT* codes,
                    const uint8_t* rt_prefixN,
                    const int* rt_parents,
                    float min_coord,
                    float range,
                    int N  // number of brt nodes
);

void k_LinkLeafNodes(OctNode* nodes,
                     const int* node_offsets,
                     const int* rt_node_counts,
                     const MortonT* codes,
                     const bool* rt_hasLeafLeft,
                     const bool* rt_hasLeafRight,
                     const uint8_t* rt_prefixN,
                     const int* rt_parents,
                     const int* rt_leftChild,
                     int N);

void checkTree(const MortonT prefix,
               int code_len,
               const OctNode* nodes,
               const int oct_idx,
               const MortonT* codes);