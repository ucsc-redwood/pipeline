
namespace gpu {

__global__ void k_PrefixSum_Deps(const int* input,
                                 int* output,
                                 const int* num_unique) {
  extern __shared__ int temp[];
  const int tid = threadIdx.x;
  int offset = 1;

  const int n = *num_unique - 1;

  temp[2 * tid] = input[2 * tid];
  temp[2 * tid + 1] = input[2 * tid + 1];

  for (int d = n >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  // Clear the last element in shared memory
  if (tid == 0) temp[n - 1] = 0;

  // Down-sweep phase
  for (int d = 1; d < n; d *= 2) {
    offset >>= 1;
    __syncthreads();
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();

  // Copy results from shared memory to global memory
  output[2 * tid] = temp[2 * tid];
  output[2 * tid + 1] = temp[2 * tid + 1];
}

}  // namespace gpu