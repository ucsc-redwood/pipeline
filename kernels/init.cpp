#include "kernels/init.hpp"

// #include <omp.h>

void k_InitDescending(unsigned int *sort, const int n) {
#pragma omp parallel for schedule(static)
  for (auto i = 0; i < n; i++) {
    sort[i] = n - i;
  }
}

void k_InitRandom(unsigned int *sort, const int n, const int seed) {
  srand(seed);

#pragma omp parallel for schedule(static)
  for (auto i = 0; i < n; i++) {
    sort[i] = rand();
  }
}

void k_InitRandomVec4(glm::vec4 *u_data,
                      const int n,
                      const float min,
                      const float range,
                      const int seed) {
  srand(seed);

#pragma omp parallel for schedule(static)
  for (auto i = 0; i < n; i++) {
    u_data[i][0] =
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * range + min;
    u_data[i][1] =
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * range + min;
    u_data[i][2] =
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * range + min;
    u_data[i][3] = 1.0f;
  }
}
