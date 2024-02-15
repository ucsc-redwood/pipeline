#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>

namespace gpu {

// Hybrid LCG-Tausworthe PRNG
// From GPU GEMS 3, Chapter 37
// Authors: Lee Howes and David Thomas
#define TAUS_STEP_1 (((z1 & 4294967294U) << 12) ^ (((z1 << 13) ^ z1) >> 19))
#define TAUS_STEP_2 (((z2 & 4294967288U) << 4) ^ (((z2 << 2) ^ z2) >> 25))
#define TAUS_STEP_3 (((z3 & 4294967280U) << 17) ^ (((z3 << 3) ^ z3) >> 11))
#define LCG_STEP (z4 * 1664525 + 1013904223U)
#define HYBRID_TAUS (z1 ^ z2 ^ z3 ^ z4)

__device__ __forceinline__ float uint_to_float(const unsigned int x) {
  return static_cast<float>(x & 0xFFFFFF) / 16777216.0f;
}

__device__ __forceinline__ float uint_to_float_scaled(const unsigned int x,
                                                      const float min,
                                                      const float range) {
  return min + uint_to_float(x) * range;
}

__global__ void k_InitRandomVec4(glm::vec4 *random_vectors,
                                 const int size,
                                 const float min,
                                 const float range,
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

    random_vectors[i] = glm::vec4(uint_to_float_scaled(z1 ^ z2, min, range),
                                  uint_to_float_scaled(z2 ^ z3, min, range),
                                  uint_to_float_scaled(z3 ^ z4, min, range),
                                  uint_to_float_scaled(z1 ^ z4, min, range));
  }
}

}  // namespace gpu