#pragma once

namespace gpu {

__global__ void k_CountUnique(unsigned int* keys, int* num_unique_out, int n);

}  // namespace gpu