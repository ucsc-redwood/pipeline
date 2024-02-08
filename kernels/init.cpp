#include "kernels/init.hpp"

#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <random>

// #if on windows, define rand_r()
#if defined(_WIN32) || defined(_WIN64)
int rand_r(unsigned int *seed) { return rand(); }
#endif

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

void k_InitRandomVec4Determinastic(glm::vec4 *u_data,
                                   const int n,
                                   const float min,
                                   const float range,
                                   const int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dis(min, min + range);

  std::generate_n(
      u_data, n, [&] { return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f); });
}

void k_InitRandomVec4(glm::vec4 *u_data,
                      const int n,
                      const float min,
                      const float range,
                      const int seed) {
  // srand(seed);
#pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    unsigned int my_seed = seed + tid;

#pragma omp parallel for schedule(static)
    for (auto i = 0; i < n; i++) {
      u_data[i][0] = static_cast<float>(rand_r(&my_seed)) /
                         static_cast<float>(RAND_MAX) * range +
                     min;
      u_data[i][1] = static_cast<float>(rand_r(&my_seed)) /
                         static_cast<float>(RAND_MAX) * range +
                     min;
      u_data[i][2] = static_cast<float>(rand_r(&my_seed)) /
                         static_cast<float>(RAND_MAX) * range +
                     min;
      u_data[i][3] = 1.0f;
    }
  }
}
