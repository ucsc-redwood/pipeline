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

const int size = 10'000'000;
const int radix = 256;
const int radixPasses = 4;
const int partitionSize = 7680;
const int globalHistPartitionSize = 65536;
const int globalHistThreads = 128;
const int binningThreads = 512;  // 2080 super seems to really like 512
const int binningThreadblocks = (size + partitionSize - 1) / partitionSize;
const int globalHistThreadblocks =
    (size + globalHistPartitionSize - 1) / globalHistPartitionSize;

void InitMemory(unsigned int* u_index,
                unsigned int* globalHistogram,
                unsigned int* firstPassHistogram,
                unsigned int* secPassHistogram,
                unsigned int* thirdPassHistogram,
                unsigned int* fourthPassHistogram) {
  cudaMemset(u_index, 0, radixPasses * sizeof(unsigned int));
  cudaMemset(globalHistogram, 0, radix * radixPasses * sizeof(unsigned int));
  cudaMemset(firstPassHistogram,
             0,
             radix * binningThreadblocks * sizeof(unsigned int));
  cudaMemset(
      secPassHistogram, 0, radix * binningThreadblocks * sizeof(unsigned int));
  cudaMemset(thirdPassHistogram,
             0,
             radix * binningThreadblocks * sizeof(unsigned int));
  cudaMemset(fourthPassHistogram,
             0,
             radix * binningThreadblocks * sizeof(unsigned int));
}

int main(const int argc, const char** argv) {
  // int n = 10'000'000;
  int n = size;
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

  OneSweepData<radixPasses> one_sweep;
  AllocManaged(&one_sweep.u_sort, n);
  AllocManaged(&one_sweep.u_sort_alt, n);
  AllocManaged(&one_sweep.u_global_histogram, radix * radixPasses);
  AllocManaged(&one_sweep.u_index, radixPasses);
  for (int i = 0; i < radixPasses; ++i) {
    AllocManaged(&one_sweep.u_pass_histograms[i], radix * binningThreadblocks);
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
    InitMemory(one_sweep.u_index,
               one_sweep.u_global_histogram,
               one_sweep.u_pass_histograms[0],
               one_sweep.u_pass_histograms[1],
               one_sweep.u_pass_histograms[2],
               one_sweep.u_pass_histograms[3]);
    checkCudaErrors(cudaDeviceSynchronize());

    spdlog::info("dispatching radix sort... with {} blocks",
                 globalHistThreadblocks);

    gpu::k_GlobalHistogram<<<globalHistThreadblocks, globalHistThreads>>>(
        one_sweep.u_sort, one_sweep.u_global_histogram, n);

    spdlog::info("dispatching k_DigitBinning... with {} blocks",
                 binningThreadblocks);

    gpu::k_DigitBinning<<<binningThreadblocks, binningThreads>>>(
        one_sweep.u_global_histogram,
        one_sweep.u_sort,
        one_sweep.u_sort_alt,
        one_sweep.u_pass_histograms[0],
        one_sweep.u_index,
        n,
        0);

    gpu::k_DigitBinning<<<binningThreadblocks, binningThreads>>>(
        one_sweep.u_global_histogram,
        one_sweep.u_sort_alt,
        one_sweep.u_sort,
        one_sweep.u_pass_histograms[1],
        one_sweep.u_index,
        n,
        8);

    gpu::k_DigitBinning<<<binningThreadblocks, binningThreads>>>(
        one_sweep.u_global_histogram,
        one_sweep.u_sort,
        one_sweep.u_sort_alt,
        one_sweep.u_pass_histograms[2],
        one_sweep.u_index,
        n,
        16);

    gpu::k_DigitBinning<<<binningThreadblocks, binningThreads>>>(
        one_sweep.u_global_histogram,
        one_sweep.u_sort_alt,
        one_sweep.u_sort,
        one_sweep.u_pass_histograms[3],
        one_sweep.u_index,
        n,
        24);

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
  for (int i = 0; i < radixPasses; ++i) {
    checkCudaErrors(cudaFree(one_sweep.u_pass_histograms[i]));
  }

  return 0;
}