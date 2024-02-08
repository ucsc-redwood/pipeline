#pragma once

#include <glm/glm.hpp>

#include "types/morton.hpp"

/* //////////////////////////////////////////////////////////////////////////////////////
 * Step 1: Convert 3D point to morton code (32-bit)
 */

namespace cpu {

MortonT single_point_to_code_v2(
    float x, float y, float z, float min_coord, float range);

void morton32_to_xyz(glm::vec4* ret,
                     MortonT code,
                     float min_coord,
                     float range);

[[maybe_unused]] int compare_uint32_t(const void* a, const void* b);

}  // namespace cpu

void k_ComputeMortonCode(const glm::vec4* data,
                         MortonT* morton_keys,
                         size_t n,
                         float min_coord,
                         float range);

namespace gpu {

void Dispatch_ComputeMortonCode_With(const glm::vec4* data,
                                     MortonT* morton_keys,
                                     size_t n,
                                     float min_coord,
                                     float range,
                                     // gpu thing
                                     int logical_num_blocks);

}