#include <cuda_runtime.h>

#include <glm/glm.hpp>

#include "types/morton.hpp"

namespace gpu {

__device__ __forceinline__ unsigned int morton3D_SplitBy3bits(
    const unsigned int a) {
  unsigned int x = ((unsigned int)a) & 0x000003ff;
  x = (x | x << 16) & 0x30000ff;
  x = (x | x << 8) & 0x0300f00f;
  x = (x | x << 4) & 0x30c30c3;
  x = (x | x << 2) & 0x9249249;
  return x;
}

__device__ __forceinline__ unsigned int m3D_e_magicbits(const unsigned int x,
                                                        const unsigned int y,
                                                        const unsigned int z) {
  return morton3D_SplitBy3bits(x) | (morton3D_SplitBy3bits(y) << 1) |
         (morton3D_SplitBy3bits(z) << 2);
}

[[nodiscard]] __device__ __forceinline__ unsigned int xyz_to_morton32(
    const glm::vec4& xyz, const float min_coord, const float range) {
  constexpr auto bit_scale = 1024;
  const auto i = static_cast<uint32_t>((xyz.x - min_coord) / range * bit_scale);
  const auto j = static_cast<uint32_t>((xyz.y - min_coord) / range * bit_scale);
  const auto k = static_cast<uint32_t>((xyz.z - min_coord) / range * bit_scale);
  return m3D_e_magicbits(i, j, k);
}

__global__ void k_ComputeMorton(const glm::vec4* d_xyz,
                                unsigned int* d_morton,
                                const int n,
                                const float min_coord,
                                const float range) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  const auto stride = blockDim.x * gridDim.x;

  for (auto i = idx; i < n; i += stride)
    d_morton[i] = xyz_to_morton32(d_xyz[i], min_coord, range);
}

// assume n = 1024
// k_ComputeMorton<<<4, 256>>>(u_data);
//   Do task...
// 
// k_ComputeMorton<<<1, 256>>>(u_data);
// for(i = 0; i < 4; ++i) {
//  Do task...
// }
//  


void Dispatch_ComputeMortonCode_With(const glm::vec4* data,
                                     MortonT* morton_keys,
                                     size_t n,
                                     float min_coord,
                                     float range,
                                     // gpu thing
                                     int logical_num_blocks) {
  constexpr auto block_size = 768;
  k_ComputeMorton<<<logical_num_blocks, block_size>>>(
      data, morton_keys, n, min_coord, range);
}

}  // namespace gpu

// void k_ComputeMortonCode(const glm::vec4* data,
//                          unsigned int* morton_keys,
//                          const size_t n,
//                          const float min_coord,
//                          const float range) {
//   gpu::k_ComputeMorton<<<1, 1>>>(data, morton_keys, n, min_coord, range);
// }
