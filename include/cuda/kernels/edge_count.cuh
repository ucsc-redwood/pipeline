#pragma once

#include <cstdint>

namespace gpu {

__global__ void k_EdgeCount_Kernel(const uint8_t* prefix_n,
                                   int* parents,
                                   int* edge_count,
                                   int n_brt_nodes);

}  // namespace gpu