#include <cuda_runtime.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <glm/glm.hpp>

#include "cuda/common/helper_cuda.hpp"
#include "kernels/all.hpp"

//
#include "cuda/constants.hpp"
#include "cuda/digit_binning.cuh"
#include "cuda/histogram.cuh"
#include "cuda/kernels/init.cuh"
#include "cuda/kernels/morton.cuh"

constexpr auto kRadix = 256;  // fixed for 32-bit unsigned int
// constexpr auto kRadixPasses = 4;

constexpr auto kLaneCount = 32;       // fixed for NVIDIA GPUs
constexpr auto kGlobalHistWarps = 8;  // configurable
constexpr auto kDigitBinWarps = 16;   // configurable

// 8x32=256 threads
const dim3 kGlobalHistDim(kLaneCount, kGlobalHistWarps, 1);

// 16x32=512 threads
const dim3 kDigitBinDim(kLaneCount, kDigitBinWarps, 1);

[[nodiscard]] constexpr int BinningThreadBlocks(
    [[maybe_unused]] const int size) {
  // looks like we want to process 15 items per thread
  // and since 512 threads was used, we have
  constexpr auto partition_size = 7680;
  return size / partition_size;
}

namespace {
void CheckHistogramCorrectness(const unsigned int* u_sort,
                               const unsigned int* u_global_histogram,
                               const int n) {
  unsigned int* u_global_histogram_ground_truth;
  checkCudaErrors(cudaMallocManaged(
      &u_global_histogram_ground_truth,
      static_cast<size_t>(kRadix) * kRadixPasses * sizeof(unsigned int)));

  k_GlobalHistogram_Original<<<2048, kGlobalHistDim>>>(
      u_sort, u_global_histogram_ground_truth, n);
  checkCudaErrors(cudaDeviceSynchronize());

  const auto is_equal = std::equal(u_global_histogram,
                                   u_global_histogram + kRadix * kRadixPasses,
                                   u_global_histogram_ground_truth);
  std::cout << "**** is_equal = " << std::boolalpha << is_equal << '\n';

  checkCudaErrors(cudaFree(u_global_histogram_ground_truth));
}
}  // namespace

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

  omp_set_num_threads(n_threads);

#pragma omp parallel
  { printf("Hello, world! I'm thread %d\n", omp_get_thread_num()); }

  constexpr auto min = 0.0f;
  constexpr auto range = 1024.0f;

  constexpr auto seed = 114514;

  glm::vec4* u_input;
  checkCudaErrors(cudaMallocManaged(&u_input, n * sizeof(glm::vec4)));
  gpu::k_InitRandomVec4(u_input, n, min, range, seed);

  checkCudaErrors(cudaDeviceSynchronize());

  unsigned int* u_sort;
  checkCudaErrors(cudaMallocManaged(&u_sort, n * sizeof(unsigned int)));

  gpu::k_ComputeMorton_Kernel<<<16, 768>>>(u_input, u_sort, n, min, range);
  checkCudaErrors(cudaDeviceSynchronize());

  unsigned int* u_sort_alt;
  checkCudaErrors(cudaMallocManaged(&u_sort_alt, n * sizeof(unsigned int)));

  // peek at the first 10 morton keys
  for (auto i = 0; i < 10; ++i) {
    spdlog::info("u_sort[{}] = {}", i, u_sort[i]);
  }

  unsigned int* u_global_histogram;
  checkCudaErrors(cudaMallocManaged(
      &u_global_histogram,
      static_cast<size_t>(kRadix) * kRadixPasses * sizeof(unsigned int)));

  unsigned int* u_index;
  checkCudaErrors(
      cudaMallocManaged(&u_index, kRadixPasses * sizeof(unsigned int)));

  std::array<unsigned int*, kRadixPasses> u_pass_histogram;

  for (auto& pass_histogram : u_pass_histogram) {
    checkCudaErrors(
        cudaMallocManaged(&pass_histogram,
                          static_cast<size_t>(kRadix) * BinningThreadBlocks(n) *
                              sizeof(unsigned int)));
  }

  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaEventRecord(start));

  k_GlobalHistogram<<<my_num_blocks, kGlobalHistDim>>>(
      u_sort, u_global_histogram, n, 2048);

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  CheckHistogramCorrectness(u_sort, u_global_histogram, n);

  float milliseconds = 0;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  std::cout << "milliseconds = " << milliseconds << '\n';

  k_DigitBinning_Original<<<my_num_blocks, kDigitBinDim>>>(u_global_histogram,
                                                           u_sort,
                                                           u_sort_alt,
                                                           u_pass_histogram[0],
                                                           u_index,
                                                           n,
                                                           0);

  k_DigitBinning_Original<<<my_num_blocks, kDigitBinDim>>>(u_global_histogram,
                                                           u_sort_alt,
                                                           u_sort,
                                                           u_pass_histogram[1],
                                                           u_index,
                                                           n,
                                                           8);

  k_DigitBinning_Original<<<my_num_blocks, kDigitBinDim>>>(u_global_histogram,
                                                           u_sort,
                                                           u_sort_alt,
                                                           u_pass_histogram[2],
                                                           u_index,
                                                           n,
                                                           16);

  k_DigitBinning_Original<<<my_num_blocks, kDigitBinDim>>>(u_global_histogram,
                                                           u_sort_alt,
                                                           u_sort,
                                                           u_pass_histogram[3],
                                                           u_index,
                                                           n,
                                                           24);

  checkCudaErrors(cudaDeviceSynchronize());

  std::cout << "done sorting\n";

  // peek 10 elements
  for (auto i = 0; i < 10; ++i) {
    std::cout << u_sort[i] << '\n';
  }

  const auto is_sorted = std::is_sorted(u_sort, u_sort + n);
  std::cout << "is_sorted = " << std::boolalpha << is_sorted << '\n';

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  checkCudaErrors(cudaFree(u_input));
  checkCudaErrors(cudaFree(u_sort));
  checkCudaErrors(cudaFree(u_sort_alt));
  checkCudaErrors(cudaFree(u_global_histogram));
  checkCudaErrors(cudaFree(u_index));
  for (const auto& pass_histogram : u_pass_histogram) {
    checkCudaErrors(cudaFree(pass_histogram));
  }
  return 0;
}