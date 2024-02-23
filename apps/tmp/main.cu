#include <algorithm>
#include <iostream>

// #include "gpu_kernels.cuh"
// #include "kernels/all.hpp"

#include "cuda/kernels/all.cuh"
#include "shared/types.h"

namespace sort {

constexpr int radix = 256;
constexpr int radixPasses = 4;

constexpr int binningThreadblocks(const int size) {
  // Need to match to the numbers in the '.cu' file in gpu source code
  constexpr int partitionSize = 7680;
  return (size + partitionSize - 1) / partitionSize;
}

constexpr int globalHistThreadblocks(const int size) {
  // Need to match to the numbers in the '.cu' file in gpu source code
  constexpr int globalHistPartitionSize = 65536;
  return (size + globalHistPartitionSize - 1) / globalHistPartitionSize;
}

}  // namespace sort

static OneSweepData<sort::radixPasses> CreateOnesweepData(const int n) {
  OneSweepData<sort::radixPasses> one_sweep;
  cudaMallocManaged(&one_sweep.u_sort, n * sizeof(unsigned int));
  cudaMallocManaged(&one_sweep.u_sort_alt, n * sizeof(unsigned int));
  cudaMallocManaged(&one_sweep.u_index,
                    sort::radixPasses * sizeof(unsigned int));
  cudaMallocManaged(&one_sweep.u_global_histogram,
                    sort::radix * sort::radixPasses * sizeof(unsigned int));
  for (int i = 0; i < sort::radixPasses; i++) {
    cudaMallocManaged(
        &one_sweep.u_pass_histograms[i],
        sort::radix * sort::binningThreadblocks(n) * sizeof(unsigned int));
  }
  return one_sweep;
}

static void DestroyOnesweepData(OneSweepData<sort::radixPasses>& one_sweep) {
  cudaFree(one_sweep.u_sort);
  cudaFree(one_sweep.u_sort_alt);
  cudaFree(one_sweep.u_index);
  cudaFree(one_sweep.u_global_histogram);
  for (int i = 0; i < sort::radixPasses; i++) {
    cudaFree(one_sweep.u_pass_histograms[i]);
  }
}

static void Dispatch_SortKernels(OneSweepData<4>& one_sweep,
                                 const int n,
                                 const int grid_size) {
  constexpr int globalHistThreads = 128;
  constexpr int binningThreads = 512;

  gpu::k_GlobalHistogram_WithLogicalBlocks<<<grid_size, globalHistThreads>>>(
      one_sweep.u_sort,
      one_sweep.u_global_histogram,
      n,
      sort::globalHistThreadblocks(n));

  gpu::k_DigitBinning_WithLogicalBlocks<<<grid_size, binningThreads>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort,
      one_sweep.u_sort_alt,
      one_sweep.u_pass_histograms[0],
      one_sweep.u_index,
      n,
      0,
      sort::binningThreadblocks(n));

  gpu::k_DigitBinning_WithLogicalBlocks<<<grid_size, binningThreads>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort_alt,
      one_sweep.u_sort,
      one_sweep.u_pass_histograms[1],
      one_sweep.u_index,
      n,
      8,
      sort::binningThreadblocks(n));

  gpu::k_DigitBinning_WithLogicalBlocks<<<grid_size, binningThreads>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort,
      one_sweep.u_sort_alt,
      one_sweep.u_pass_histograms[2],
      one_sweep.u_index,
      n,
      16,
      sort::binningThreadblocks(n));

  gpu::k_DigitBinning_WithLogicalBlocks<<<grid_size, binningThreads>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort_alt,
      one_sweep.u_sort,
      one_sweep.u_pass_histograms[3],
      one_sweep.u_index,
      n,
      24,
      sort::binningThreadblocks(n));
}

int main(const int argc, const char* argv[]) {
  constexpr auto n = 1920 * 1080;

  int grid_size = 64;

  if (argc > 1) {
    grid_size = std::strtol(argv[1], nullptr, 10);
  }

  std::cout << "Grid size: " << grid_size << std::endl;

  OneSweepData<4> one_sweep;
  cudaMallocManaged(&one_sweep.u_sort, n * sizeof(unsigned int));
  cudaMallocManaged(&one_sweep.u_sort_alt, n * sizeof(unsigned int));
  cudaMallocManaged(&one_sweep.u_index,
                    sort::radixPasses * sizeof(unsigned int));
  cudaMallocManaged(&one_sweep.u_global_histogram,
                    sort::radix * sort::radixPasses * sizeof(unsigned int));
  for (int i = 0; i < 4; i++) {
    cudaMallocManaged(
        &one_sweep.u_pass_histograms[i],
        sort::radix * sort::binningThreadblocks(n) * sizeof(unsigned int));
  }

  std::generate_n(one_sweep.u_sort, n, [n = n]() mutable { return --n; });

  auto is_sorted = std::is_sorted(one_sweep.u_sort, one_sweep.u_sort + n);
  std::cout << "Is sorted: " << std::boolalpha << is_sorted << std::endl;

  Dispatch_SortKernels(one_sweep, n, grid_size);
  cudaDeviceSynchronize();

  is_sorted = std::is_sorted(one_sweep.u_sort, one_sweep.u_sort + n);
  std::cout << "Is sorted: " << std::boolalpha << is_sorted << std::endl;

  cudaFree(one_sweep.u_sort);
  cudaFree(one_sweep.u_sort_alt);
  cudaFree(one_sweep.u_index);
  cudaFree(one_sweep.u_global_histogram);
  for (int i = 0; i < 4; i++) {
    cudaFree(one_sweep.u_pass_histograms[i]);
  }

  return 0;
}