#pragma once

namespace gpu {

// need to create 2*block_size shared memory
__global__ void k_PrefixSum_Deps(const int* in, int* out, const int* n);

}