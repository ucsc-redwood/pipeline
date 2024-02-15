#include <device_launch_parameters.h>

namespace gpu {

__global__ void k_CountUnique(unsigned int* keys,
                              int* num_unique_out,
                              const int n) {
  if (const auto tid = threadIdx.x + blockIdx.x * blockDim.x; tid == 0) {
    if (n == 0) {
      *num_unique_out = 0;
      return;
    }

    auto j = 0;
    for (auto i = 1; i < n; ++i) {
      if (keys[i] != keys[j]) {
        ++j;
        keys[j] = keys[i];
      }
    }

    *num_unique_out = j + 1;
  }
}

}  // namespace gpu