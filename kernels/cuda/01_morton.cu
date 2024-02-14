#include <cuda_runtime.h>

#include "shared/morton.h"

namespace gpu {

__global__ void k_ComputeMortonCode(const glm::vec4* data,
                                    unsigned int* morton_keys,
                                    const size_t n,
                                    const float min_coord,
                                    const float range) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  const auto stride = blockDim.x * gridDim.x;

  for (auto i = idx; i < n; i += stride) {
    morton_keys[i] = shared::xyz_to_morton32(data[i], min_coord, range);
  }
}

}  // namespace gpu