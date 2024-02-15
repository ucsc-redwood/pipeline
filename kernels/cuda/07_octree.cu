#include <device_launch_parameters.h>

#include "cuda/kernels/07_octree.cuh"
#include "shared/oct.h"

namespace gpu {

__global__ void k_MakeOctNodes(OctNode* oct_nodes,
                               const int* node_offsets,    // prefix sum
                               const int* rt_node_counts,  // edge count
                               const unsigned int* codes,
                               const uint8_t* rt_prefixN,
                               const int* rt_parents,
                               const float min_coord,
                               const float range,
                               const int N  // number of brt nodes
) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  const auto stride = blockDim.x * gridDim.x;

  // i > 0 && i < N
  for (auto i = idx; i < N; i += stride) {
    if (i == 0) {
      continue;
    }

    shared::ProcessOctNode(i,
                           oct_nodes,
                           node_offsets,
                           rt_node_counts,
                           codes,
                           rt_prefixN,
                           rt_parents,
                           min_coord,
                           range,
                           N);
  }
}

__global__ void k_LinkLeafNodes(OctNode* nodes,
                                const int* node_offsets,
                                const int* rt_node_counts,
                                const unsigned int* codes,
                                const bool* rt_hasLeafLeft,
                                const bool* rt_hasLeafRight,
                                const uint8_t* rt_prefixN,
                                const int* rt_parents,
                                const int* rt_leftChild,
                                const int N) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  const auto stride = blockDim.x * gridDim.x;

  // i < N
  for (auto i = idx; i < N; i += stride) {
    shared::ProcessLinkLeaf(i,
                            nodes,
                            node_offsets,
                            rt_node_counts,
                            codes,
                            rt_hasLeafLeft,
                            rt_hasLeafRight,
                            rt_prefixN,
                            rt_parents,
                            rt_leftChild,
                            N);
  }
}

}  // namespace gpu