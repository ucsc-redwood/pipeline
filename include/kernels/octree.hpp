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