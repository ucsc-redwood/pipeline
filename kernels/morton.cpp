#include "kernels/morton.hpp"

#include <omp.h>

MortonT single_point_to_code_v2(
    float x, float y, float z, const float min_coord, const float range) {
  const float bit_scale = 1024.0f;

  x = (x - min_coord) / range;
  y = (y - min_coord) / range;
  z = (z - min_coord) / range;

  return m3D_e_magicbits(static_cast<CoordT>(x * bit_scale),
                         static_cast<CoordT>(y * bit_scale),
                         static_cast<CoordT>(z * bit_scale));
}

void morton32_to_xyz(glm::vec4* ret,
                     const MortonT code,
                     const float min_coord,
                     const float range) {
  const float bit_scale = 1024.0f;

  CoordT dec_raw_x[3];
  m3D_d_magicbits(code, dec_raw_x);

  const float dec_x = ((float)dec_raw_x[0] / bit_scale) * range + min_coord;
  const float dec_y = ((float)dec_raw_x[1] / bit_scale) * range + min_coord;
  const float dec_z = ((float)dec_raw_x[2] / bit_scale) * range + min_coord;

  // vec4 result = {dec_x, dec_y, dec_z, 1.0f};
  // glm_vec4_copy(result, *ret);
  (*ret)[0] = dec_x;
  (*ret)[1] = dec_y;
  (*ret)[2] = dec_z;
  (*ret)[3] = 1.0f;
}

// functor for uint32_t, used in qsort
int compare_uint32_t(const void* a, const void* b) {
  const unsigned int value1 = *(const unsigned int*)a;
  const unsigned int value2 = *(const unsigned int*)b;

  if (value1 < value2) return -1;
  if (value1 > value2) return 1;
  return 0;
}

void k_ComputeMortonCode(const glm::vec4* data,
                         unsigned int* morton_keys,
                         const size_t n,
                         const float min_coord,
                         const float range) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
    morton_keys[i] = single_point_to_code_v2(
        data[i][0], data[i][1], data[i][2], min_coord, range);
  }
}
