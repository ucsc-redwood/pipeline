#pragma once

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <stdexcept>

#define BENCH_CUDA_TRY(call)                                \
  do {                                                      \
    auto const status = (call);                             \
    if (cudaSuccess != status) {                            \
      throw std::runtime_error(cudaGetErrorString(status)); \
    }                                                       \
  } while (0);

template <class T>
[[nodiscard]] static int DetermineBlockSize(T func) {
  int blockSize = 1;
  int minGridSize = 1;
  BENCH_CUDA_TRY(
      cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func));
  return blockSize;
}

template <class T>
[[nodiscard]] static int DetermineBlockSizeAndDisplay(T func,
                                                      benchmark::State &state) {
  const auto block_size = DetermineBlockSize(func);
  state.counters["block_size"] = block_size;
  return block_size;
}

template <typename T>
[[nodiscard]] static T *AllocateDevice(const int n) {
  T *d_data;
  BENCH_CUDA_TRY(cudaMalloc(&d_data, n * sizeof(T)));
  return d_data;
}

template <typename T>
[[nodiscard]] static T *AllocateHost(const int n) {
  T *h_data;
  BENCH_CUDA_TRY(cudaMallocHost(&h_data, n * sizeof(T)));
  return h_data;
}

template <typename T>
[[nodiscard]] static T *AllocateManaged(const int n) {
  T *u_data;
  BENCH_CUDA_TRY(cudaMallocManaged(&u_data, n * sizeof(T)));
  return u_data;
}

static void Free(void *ptr) { BENCH_CUDA_TRY(cudaFree(ptr)); }

class CudaEventTimer {
 public:
  /**
   * @brief Constructs a `CudaEventTimer` beginning a manual timing range.
   *
   * Optionally flushes L2 cache.
   *
   * @param[in,out] state  This is the benchmark::State whose timer we are going
   * to update.
   * @param[in] flush_l2_cache_ whether or not to flush the L2 cache before
   *                            every iteration.
   * @param[in] stream_ The CUDA stream we are measuring time on.
   */
  CudaEventTimer(benchmark::State &state,
                 const bool flush_l2_cache = false,
                 const cudaStream_t stream = 0)
      : p_state(&state), stream_(stream) {
    // flush all of L2$
    if (flush_l2_cache) {
      int current_device = 0;
      BENCH_CUDA_TRY(cudaGetDevice(&current_device));

      int l2_cache_bytes = 0;
      BENCH_CUDA_TRY(cudaDeviceGetAttribute(
          &l2_cache_bytes, cudaDevAttrL2CacheSize, current_device));

      if (l2_cache_bytes > 0) {
        const int memset_value = 0;
        int *l2_cache_buffer = nullptr;
        BENCH_CUDA_TRY(cudaMalloc(&l2_cache_buffer, l2_cache_bytes));
        BENCH_CUDA_TRY(cudaMemsetAsync(
            l2_cache_buffer, memset_value, l2_cache_bytes, stream_));
        BENCH_CUDA_TRY(cudaFree(l2_cache_buffer));
      }
    }

    BENCH_CUDA_TRY(cudaEventCreate(&start_));
    BENCH_CUDA_TRY(cudaEventCreate(&stop_));
    BENCH_CUDA_TRY(cudaEventRecord(start_, stream_));
  }

  CudaEventTimer() = delete;

  /**
   * @brief Destroy the `CudaEventTimer` and ending the manual time range.
   *
   */
  ~CudaEventTimer() {
    BENCH_CUDA_TRY(cudaEventRecord(stop_, stream_));
    BENCH_CUDA_TRY(cudaEventSynchronize(stop_));
    float milliseconds = 0.0f;
    BENCH_CUDA_TRY(cudaEventElapsedTime(&milliseconds, start_, stop_));
    p_state->SetIterationTime(milliseconds / (1000.0f));
    BENCH_CUDA_TRY(cudaEventDestroy(start_));
    BENCH_CUDA_TRY(cudaEventDestroy(stop_));
  }

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
  cudaStream_t stream_;
  benchmark::State *p_state;
};
