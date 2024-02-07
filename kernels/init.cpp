#include "kernels/init.hpp"

#include <omp.h>

void k_InitDescending(unsigned int *sort, size_t n) {
#pragma omp parallel for schedule(static)
  for (auto i = 0; i < n; i++) {
    sort[i] = n - i;
  }
}

void k_InitRandom(unsigned int *sort, size_t n, int seed) {
  srand(seed);

#pragma omp parallel for schedule(static)
  for (auto i = 0; i < n; i++) {
    sort[i] = rand();
  }
}

void k_InitRandomVec4(
    glm::vec4 *u_data, size_t n, float min, float range, int seed) {
  srand(seed);

#pragma omp parallel for schedule(static)
  for (auto i = 0; i < n; i++) {
    u_data[i][0] = (float)rand() / (float)RAND_MAX * range + min;
    u_data[i][1] = (float)rand() / (float)RAND_MAX * range + min;
    u_data[i][2] = (float)rand() / (float)RAND_MAX * range + min;
    u_data[i][3] = 1.0f;
  }
}
