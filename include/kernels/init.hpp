#pragma once

#include <glm/glm.hpp>

void k_InitDescending(unsigned int *sort, size_t n);

void k_InitRandom(unsigned int *sort, size_t n, int seed);

void k_InitRandomVec4(
    glm::vec4 *u_data, size_t n, float min, float range, int seed);
