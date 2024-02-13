#pragma once

#include <glm/glm.hpp>

void k_InitDescending(unsigned int *sort, int n);

void k_InitAscending(unsigned int *sort, int n);

// squential to make sure the same result
void k_InitRandomVec4(
    glm::vec4 *u_data, int n, float min, float range, int seed);
