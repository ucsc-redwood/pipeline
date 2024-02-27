#pragma once

#include <cstdint>

namespace gpu {

[[deprecated]] __global__ void k_EdgeCount(const uint8_t* prefix_n,
                                           const int* parents,
                                           int* edge_count,
                                           int n_brt_nodes);

// Intentially change the nbrt_nodes to a pointer, so we can chain the kernels
// and avoid doing any thing in the host code.
__global__ void k_EdgeCount_Deps(const uint8_t* prefix_n,
                                 const int* parents,
                                 int* edge_count,
                                 const int* n_unique);

}  // namespace gpu
