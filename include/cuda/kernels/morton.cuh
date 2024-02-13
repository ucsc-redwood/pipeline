#pragma once

namespace gpu {

__global__ void k_ComputeMorton_Kernel(const glm::vec4* d_xyz,
                                       unsigned int* morton_keys,
                                       const int n,
                                       const float min_coord,
                                       const float range);

}  // namespace gpu