#pragma once

#include <glm/glm.hpp>

void k_InitDescending(unsigned int *sort, int n);

void k_InitRandomVec4(
    glm::vec4 *u_data, int n, float min, float range, int seed);

namespace gpu {

void k_InitRandomVec4(
    glm::vec4 *u_data, int n, float min, float range, int seed);

void k_InitAscendingSync(unsigned int *sort, int n);

}  // namespace gpu