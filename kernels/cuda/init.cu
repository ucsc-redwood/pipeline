#include <glm/glm.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace gpu {

// Hybrid LCG-Tausworthe PRNG
// From GPU GEMS 3, Chapter 37
// Authors: Lee Howes and David Thomas
#define TAUS_STEP_1 (((z1 & 4294967294U) << 12) ^ (((z1 << 13) ^ z1) >> 19))
#define TAUS_STEP_2 (((z2 & 4294967288U) << 4) ^ (((z2 << 2) ^ z2) >> 25))
#define TAUS_STEP_3 (((z3 & 4294967280U) << 17) ^ (((z3 << 3) ^ z3) >> 11))
#define LCG_STEP (z4 * 1664525 + 1013904223U)
#define HYBRID_TAUS (z1 ^ z2 ^ z3 ^ z4)

__host__ __device__ __forceinline__ float uintToFloat(unsigned int x) {
  return (x & 0xFFFFFF) / 16777216.0f;
}

__host__ __device__ __forceinline__ float uintToFloatScaled(unsigned int x) {
  return uintToFloat(x) * 1024.0f;
}

__global__ void k_InitRandomVec4_Kernel(glm::vec4 *random_vectors,
                                        const int size,
                                        const int seed) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  unsigned int z1 = (idx << 2) * seed;
  unsigned int z2 = ((idx << 2) + 1) * seed;
  unsigned int z3 = ((idx << 2) + 2) * seed;
  unsigned int z4 = ((idx << 2) + 3) * seed;

  for (auto i = idx; i < size; i += blockDim.x * gridDim.x) {
    z1 = TAUS_STEP_1;
    z2 = TAUS_STEP_2;
    z3 = TAUS_STEP_3;
    z4 = LCG_STEP;

    random_vectors[i] = glm::vec4(uintToFloatScaled(z1 ^ z2),
                                  uintToFloatScaled(z2 ^ z3),
                                  uintToFloatScaled(z3 ^ z4),
                                  uintToFloatScaled(z1 ^ z4));
  }
}

void k_InitRandomVec4(
    glm::vec4 *u_data, int n, float min, float range, int seed) {
  // I tested this using the API and it was the fastest
  constexpr auto block_size = 768;
  const auto grid_size = (n + block_size - 1) / block_size;
  k_InitRandomVec4_Kernel<<<grid_size, block_size>>>(u_data, n, seed);
}

}  // namespace gpu