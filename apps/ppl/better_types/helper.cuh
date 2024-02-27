#pragma once

// if in release mode, we don't want to check for cuda errors
#ifdef NDEBUG
#define CHECK_CUDA_CALL(x) x
#else
#include "cuda/common/helper_cuda.hpp"
#define CHECK_CUDA_CALL(x) checkCudaErrors(x)
#endif

template <typename T>
constexpr void MallocManaged(T** ptr, const size_t num_items) {
  CHECK_CUDA_CALL(
      cudaMallocManaged(reinterpret_cast<void**>(ptr), num_items * sizeof(T)));
}

#define CUDA_FREE(ptr) CHECK_CUDA_CALL(cudaFree(ptr))

#define ATTACH_STREAM_SINGLE(ptr) \
  CHECK_CUDA_CALL(cudaStreamAttachMemAsync(stream, ptr, 0, cudaMemAttachSingle))
