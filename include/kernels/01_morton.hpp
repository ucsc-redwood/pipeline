#pragma once

#include <glm/glm.hpp>

/* //////////////////////////////////////////////////////////////////////////////////////
 * Step 1: Convert 3D point to morton code (32-bit)
 */

void k_ComputeMortonCode(const glm::vec4* data,
                         unsigned int* morton_keys,
                         size_t n,
                         float min_coord,
                         float range);
