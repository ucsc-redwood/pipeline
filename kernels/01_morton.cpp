#include "kernels/01_morton.hpp"

#include "shared/morton.h"

void k_ComputeMortonCode(const glm::vec4* data,
                         unsigned int* morton_keys,
                         const size_t n,
                         const float min_coord,
                         const float range) {
#pragma omp parallel for
  for (auto i = 0; i < n; i++) {
    morton_keys[i] = shared::xyz_to_morton32(data[i], min_coord, range);
  }
}
