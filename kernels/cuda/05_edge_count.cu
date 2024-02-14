#include "cuda/kernels/05_edge_count.cuh"

namespace gpu {

__global__ void k_EdgeCount(const uint8_t* prefix_n,
                            const int* parents,
                            int* edge_count,
                            const int n_brt_nodes) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  const auto stride = blockDim.x * gridDim.x;

  for (auto i = idx; i < n_brt_nodes; i += stride) {
    if (i == 0) {
      edge_count[i] = 0;
      continue;
    }
    const auto my_depth = prefix_n[i] / 3;
    const auto parent_depth = prefix_n[parents[i]] / 3;
    edge_count[i] = my_depth - parent_depth;
  }
}

}  // namespace gpu