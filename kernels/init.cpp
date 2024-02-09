#include "kernels/init.hpp"

#include <algorithm>
#include <random>

void k_InitDescending(unsigned int* sort, const int n) {
#pragma omp parallel for
  for (auto i = 0; i < n; i++) {
    sort[i] = n - i;
  }
}

void k_InitRandomVec4(glm::vec4* u_data,
                      const int n,
                      const float min,
                      const float range,
                      const int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution dis(min, min + range);

  std::generate_n(
      u_data, n, [&] { return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f); });
}
