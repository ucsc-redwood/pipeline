#include <cuda_runtime.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <glm/glm.hpp>

#include "cuda/common/helper_cuda.hpp"
#include "cuda/kernels/all.cuh"
#include "shared/types.h"

template <typename T>
[[nodiscard]] T* AllocManaged(const int n) {
  T* ptr;
  checkCudaErrors(cudaMallocManaged(&ptr, n * sizeof(T)));
  return ptr;
}

template <typename T>
void AllocManaged(T** ptr, const int n) {
  checkCudaErrors(cudaMallocManaged(ptr, n * sizeof(T)));
}

constexpr auto kRadix = RADIX;
constexpr auto kLaneCount = LANE_COUNT;          // fixed for NVIDIA GPUs: 32
constexpr auto kGlobalHistWarps = G_HIST_WARPS;  // configurable: 8
constexpr auto kDigitBinWarps = BIN_WARPS;       // configurable: 16


// 8x32=256 threads
const dim3 kGlobalHistDim(kLaneCount, kGlobalHistWarps, 1);

// 16x32=512 threads
const dim3 kDigitBinDim(kLaneCount, kDigitBinWarps, 1);

[[nodiscard]] constexpr int GlobalHistThreadBlocks(const int size){
  constexpr int globalHistPartitionSize = 65536;
  
}

[[nodiscard]] constexpr int BinningThreadBlocks(
    [[maybe_unused]] const int size) {
  // looks like we want to process 15 items per thread
  // and since 512 threads was used, we have 15*512 = 7680
  constexpr auto partition_size = 7680;
  return (size + partition_size - 1) / partition_size;
}

int main(const int argc, const char** argv) {
  int n = 10'000'000;
  int n_threads = 4;
  int my_num_blocks = 64;

  CLI::App app{"Multi-threaded sorting benchmark"};

  app.add_option("-n,--n", n, "Number of elements to sort")
      ->check(CLI::PositiveNumber);

  app.add_option("-t,--threads", n_threads, "Number of threads to use")
      ->check(CLI::Range(1, 48));

  app.add_option("-b,--blocks", my_num_blocks, "Number of blocks to use")
      ->check(CLI::PositiveNumber);

  CLI11_PARSE(app, argc, argv)

  spdlog::info("n = {}", n);
  spdlog::info("n_threads = {}", n_threads);
  spdlog::info("my_num_blocks = {}", my_num_blocks);

  omp_set_num_threads(n_threads);

  // ---------------------------------------------------------------------------

  constexpr auto min = 0.0f;
  constexpr auto max = 1024.0f;
  constexpr auto range = max - min;

  auto u_data = AllocManaged<glm::vec4>(n);

  OneSweepData one_sweep;
  AllocManaged(&one_sweep.u_sort, n);
  AllocManaged(&one_sweep.u_sort_alt, n);
  AllocManaged(&one_sweep.u_global_histogram, RADIX * kRadixPasses);
  AllocManaged(&one_sweep.u_index, kRadixPasses);
  for (int i = 0; i < kRadixPasses; ++i) {
    AllocManaged(&one_sweep.u_pass_histograms[i], RADIX * kRadixPasses);
  }

  {
    constexpr auto num_threads = 768;
    constexpr auto seed = 114514;
    const auto grid_size = (n + num_threads - 1) / num_threads;
    gpu::k_InitRandomVec4<<<grid_size, num_threads>>>(
        u_data, n, min, range, seed);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  // peek 10 elements
  for (int i = 0; i < 10; ++i) {
    spdlog::debug("u_data[{}] = ({}, {}, {}, {})",
                  i,
                  u_data[i].x,
                  u_data[i].y,
                  u_data[i].z,
                  u_data[i].w);
  }

  {
    constexpr auto num_threads = 768;
    gpu::k_ComputeMortonCode<<<my_num_blocks, num_threads>>>(
        u_data, one_sweep.u_sort, n, min, range);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  // peek 10 elements
  for (int i = 0; i < 10; ++i) {
    spdlog::debug("one_sweep.u_sort[{}] = {}", i, one_sweep.u_sort[i]);
  }

  // Sorting kernels
  {
    spdlog::info("dispatching radix sort... with {} blocks", my_num_blocks);

    gpu::k_GlobalHistogram<<<my_num_blocks, kGlobalHistDim>>>(
        one_sweep.u_sort, one_sweep.u_global_histogram, n, 2048);

    spdlog::info("dispatching radix sort... with {} blocks",
                 BinningThreadBlocks(n));

    gpu::k_DigitBinning<<<BinningThreadBlocks(n), kDigitBinDim>>>(
        one_sweep.u_global_histogram,
        one_sweep.u_sort,
        one_sweep.u_sort_alt,
        one_sweep.u_pass_histograms[0],
        one_sweep.u_index,
        n,
        0);

    // gpu::k_DigitBinning<<<BinningThreadBlocks(n), kDigitBinDim>>>(
    //     one_sweep.u_global_histogram,
    //     one_sweep.u_sort_alt,
    //     one_sweep.u_sort,
    //     one_sweep.u_pass_histograms[1],
    //     one_sweep.u_index,
    //     n,
    //     8);

    // gpu::k_DigitBinning<<<BinningThreadBlocks(n), kDigitBinDim>>>(
    //     one_sweep.u_global_histogram,
    //     one_sweep.u_sort,
    //     one_sweep.u_sort_alt,
    //     one_sweep.u_pass_histograms[2],
    //     one_sweep.u_index,
    //     n,
    //     16);

    // gpu::k_DigitBinning<<<BinningThreadBlocks(n), kDigitBinDim>>>(
    //     one_sweep.u_global_histogram,
    //     one_sweep.u_sort_alt,
    //     one_sweep.u_sort,
    //     one_sweep.u_pass_histograms[3],
    //     one_sweep.u_index,
    //     n,
    //     24);

    checkCudaErrors(cudaDeviceSynchronize());
  }

  spdlog::info("Done Sorting!");

  // peek 10 elements
  for (int i = 0; i < 10; ++i) {
    spdlog::info("one_sweep.u_sort[{}] = {}", i, one_sweep.u_sort[i]);
  }

  const auto is_sorted = std::is_sorted(one_sweep.u_sort, one_sweep.u_sort + n);
  std::cout << "is_sorted = " << std::boolalpha << is_sorted << '\n';

  checkCudaErrors(cudaFree(u_data));
  checkCudaErrors(cudaFree(one_sweep.u_sort));
  checkCudaErrors(cudaFree(one_sweep.u_sort_alt));
  checkCudaErrors(cudaFree(one_sweep.u_global_histogram));
  checkCudaErrors(cudaFree(one_sweep.u_index));
  for (int i = 0; i < kRadixPasses; ++i) {
    checkCudaErrors(cudaFree(one_sweep.u_pass_histograms[i]));
  }

  return 0;
}