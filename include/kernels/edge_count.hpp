#pragma once

#include <cstdint>

void k_EdgeCount(const uint8_t* prefix_n,
                 int* parents,
                 int* edge_count,
                 int n_brt_nodes);