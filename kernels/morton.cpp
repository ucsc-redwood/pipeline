#include "kernels/morton.hpp"

namespace cpu {

// Thus the maximum depth of the octree is 10 for 32-bit morton code.
// for 32-bit morton, each 3-bit is used to encode one coordinate.
// we can only use 10 chunk of 3-bits, so 2 bits are wasted.
// for 64-bit morton,
// we can use 21 chunk of 3-bits, so 63. 1 bit is wasted.
// enum { MORTON_BITS = 30 };

static MortonT morton3D_SplitBy3bits(const CoordT a) {
  MortonT x = static_cast<MortonT>(a) & 0x000003ff;
  x = (x | x << 16) & 0x30000ff;
  x = (x | x << 8) & 0x0300f00f;
  x = (x | x << 4) & 0x30c30c3;
  x = (x | x << 2) & 0x9249249;
  return x;
}

static MortonT m3D_e_magicbits(const CoordT x, const CoordT y, const CoordT z) {
  return morton3D_SplitBy3bits(x) | (morton3D_SplitBy3bits(y) << 1) |
         (morton3D_SplitBy3bits(z) << 2);
}

static CoordT morton3D_GetThirdBits(const MortonT m) {
  MortonT x = m & 0x9249249;
  x = (x ^ (x >> 2)) & 0x30c30c3;
  x = (x ^ (x >> 4)) & 0x0300f00f;
  x = (x ^ (x >> 8)) & 0x30000ff;
  x = (x ^ (x >> 16)) & 0x000003ff;
  return x;
}

static void m3D_d_magicbits(const MortonT m, CoordT* xyz) {
  xyz[0] = morton3D_GetThirdBits(m);
  xyz[1] = morton3D_GetThirdBits(m >> 1);
  xyz[2] = morton3D_GetThirdBits(m >> 2);
}

MortonT single_point_to_code_v2(
    float x, float y, float z, const float min_coord, const float range) {
  constexpr auto bit_scale = 1024.0f;

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
  constexpr auto bit_scale = 1024.0f;

  CoordT dec_raw_x[3];
  m3D_d_magicbits(code, dec_raw_x);

  const auto dec_x =
      (static_cast<float>(dec_raw_x[0]) / bit_scale) * range + min_coord;
  const auto dec_y =
      (static_cast<float>(dec_raw_x[1]) / bit_scale) * range + min_coord;
  const auto dec_z =
      (static_cast<float>(dec_raw_x[2]) / bit_scale) * range + min_coord;

  // vec4 result = {dec_x, dec_y, dec_z, 1.0f};
  // glm_vec4_copy(result, *ret);
  (*ret)[0] = dec_x;
  (*ret)[1] = dec_y;
  (*ret)[2] = dec_z;
  (*ret)[3] = 1.0f;
}

// functor for uint32_t, used in qsort
int compare_uint32_t(const void* a, const void* b) {
  const auto value1 = *static_cast<const unsigned int*>(a);
  const auto value2 = *static_cast<const unsigned int*>(b);

  if (value1 < value2) return -1;
  if (value1 > value2) return 1;
  return 0;
}

}  // namespace cpu

void k_ComputeMortonCode(const glm::vec4* data,
                         unsigned int* morton_keys,
                         const size_t n,
                         const float min_coord,
                         const float range) {
#pragma omp parallel for
  for (auto i = 0; i < n; i++) {
    morton_keys[i] = cpu::single_point_to_code_v2(
        data[i][0], data[i][1], data[i][2], min_coord, range);
  }
}
