#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <cuda/common/helper_cuda.hpp>

namespace gpu {
void k_CubRadixSort(unsigned int *keys, unsigned int *keys_alt, int n) {
  size_t temp_storage_bytes = 0;
  void *d_temp_storage = nullptr;

  cub::DeviceRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, keys, keys_alt, n);

  checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  cub::DeviceRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, keys, keys_alt, n);

  checkCudaErrors(cudaFree(d_temp_storage));
}
}  // namespace gpu