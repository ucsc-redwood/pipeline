#pragma once

#include <glm/glm.hpp>

// Thus the maximum depth of the octree is 10 for 32-bit morton code.
// for 32-bit morton, each 3-bit is used to encode one coordinate.
// we can only use 10 chunk of 3-bits, so 2 bits are wasted.
// for 64-bit morton,
// we can use 21 chunk of 3-bits, so 63. 1 bit is wasted.
// enum { MORTON_BITS = 30 };

constexpr auto MORTON_BITS = 30;

using MortonT = unsigned int;
using CoordT = unsigned int;

static MortonT morton3D_SplitBy3bits(const CoordT a) {
  MortonT x = ((MortonT)a) & 0x000003ff;
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

static inline void m3D_d_magicbits(const MortonT m, CoordT* xyz) {
  xyz[0] = morton3D_GetThirdBits(m);
  xyz[1] = morton3D_GetThirdBits(m >> 1);
  xyz[2] = morton3D_GetThirdBits(m >> 2);
}

/* //////////////////////////////////////////////////////////////////////////////////////
 * Step 1: Convert 3D point to morton code (32-bit)
 */

MortonT single_point_to_code_v2(
    float x, float y, float z, float min_coord, float range);

void morton32_to_xyz(glm::vec4* ret,
                     MortonT code,
                     float min_coord,
                     float range);

int compare_uint32_t(const void* a, const void* b);

void k_ComputeMortonCode(const glm::vec4* data,
                         MortonT* morton_keys,
                         size_t n,
                         float min_coord,
                         float range);
