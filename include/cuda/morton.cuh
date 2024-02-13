#pragma once

namespace gpu {

// ---------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------

__host__ __device__ __forceinline__ unsigned int morton3D_SplitBy3bits(
    const unsigned int a) {
  auto x = static_cast<unsigned int>(a) & 0x000003ff;
  x = (x | x << 16) & 0x30000ff;
  x = (x | x << 8) & 0x0300f00f;
  x = (x | x << 4) & 0x30c30c3;
  x = (x | x << 2) & 0x9249249;
  return x;
}

__host__ __device__ __forceinline__ unsigned int m3D_e_magicbits(
    const unsigned int x, const unsigned int y, const unsigned int z) {
  return morton3D_SplitBy3bits(x) | (morton3D_SplitBy3bits(y) << 1) |
         (morton3D_SplitBy3bits(z) << 2);
}

[[nodiscard]] __host__ __device__ __forceinline__ unsigned int xyz_to_morton32(
    const glm::vec4& xyz, const float min_coord, const float range) {
  constexpr auto bit_scale = 1024;
  const auto i = static_cast<uint32_t>((xyz.x - min_coord) / range * bit_scale);
  const auto j = static_cast<uint32_t>((xyz.y - min_coord) / range * bit_scale);
  const auto k = static_cast<uint32_t>((xyz.z - min_coord) / range * bit_scale);
  return m3D_e_magicbits(i, j, k);
}

// ---------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------

__host__ __device__ __forceinline__ unsigned int morton3D_GetThirdBits(
    const unsigned int m) {
  auto x = m & 0x9249249;
  x = (x ^ (x >> 2)) & 0x30c30c3;
  x = (x ^ (x >> 4)) & 0x0300f00f;
  x = (x ^ (x >> 8)) & 0x30000ff;
  x = (x ^ (x >> 16)) & 0x000003ff;
  return x;
}

__host__ __device__ __forceinline__ void m3D_d_magicbits(const unsigned int m,
                                                         unsigned int* xyz) {
  xyz[0] = morton3D_GetThirdBits(m);
  xyz[1] = morton3D_GetThirdBits(m >> 1);
  xyz[2] = morton3D_GetThirdBits(m >> 2);
}

__host__ __device__ __forceinline__ void morton32_to_xyz(
    glm::vec4* ret,
    const unsigned int code,
    const float min_coord,
    const float range) {
  constexpr auto bit_scale = 1024.0f;

  unsigned int dec_raw_x[3];
  m3D_d_magicbits(code, dec_raw_x);

  const auto dec_x =
      (static_cast<float>(dec_raw_x[0]) / bit_scale) * range + min_coord;
  const auto dec_y =
      (static_cast<float>(dec_raw_x[1]) / bit_scale) * range + min_coord;
  const auto dec_z =
      (static_cast<float>(dec_raw_x[2]) / bit_scale) * range + min_coord;

  (*ret)[0] = dec_x;
  (*ret)[1] = dec_y;
  (*ret)[2] = dec_z;
  (*ret)[3] = 1.0f;
}

}  // namespace gpu