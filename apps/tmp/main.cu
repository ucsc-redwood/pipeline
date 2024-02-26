#include <algorithm>
#include <iostream>
#include <random>

#include "cuda/kernels/02_sort.cuh"
#include "data_packs.cuh"

inline void Dispatch_SortKernels(OneSweep& one_sweep,
                                 const int n,
                                 const int grid_size,
                                 const cudaStream_t& stream) {
  constexpr int globalHistThreads = 128;
  constexpr int binningThreads = 512;

  gpu::k_GlobalHistogram_WithLogicalBlocks<<<grid_size,
                                             globalHistThreads,
                                             0,
                                             stream>>>(
      one_sweep.u_sort,
      one_sweep.u_global_histogram,
      n,
      OneSweep::globalHistThreadblocks(n));

  gpu::k_DigitBinning_WithLogicalBlocks<<<grid_size,
                                          binningThreads,
                                          0,
                                          stream>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort,
      one_sweep.u_sort_alt,
      one_sweep.u_pass_histograms[0],
      one_sweep.u_index,
      n,
      0,
      OneSweep::binningThreadblocks(n));

  gpu::k_DigitBinning_WithLogicalBlocks<<<grid_size,
                                          binningThreads,
                                          0,
                                          stream>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort_alt,
      one_sweep.u_sort,
      one_sweep.u_pass_histograms[1],
      one_sweep.u_index,
      n,
      8,
      OneSweep::binningThreadblocks(n));

  gpu::k_DigitBinning_WithLogicalBlocks<<<grid_size,
                                          binningThreads,
                                          0,
                                          stream>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort,
      one_sweep.u_sort_alt,
      one_sweep.u_pass_histograms[2],
      one_sweep.u_index,
      n,
      16,
      OneSweep::binningThreadblocks(n));

  gpu::k_DigitBinning_WithLogicalBlocks<<<grid_size,
                                          binningThreads,
                                          0,
                                          stream>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort_alt,
      one_sweep.u_sort,
      one_sweep.u_pass_histograms[3],
      one_sweep.u_index,
      n,
      24,
      OneSweep::binningThreadblocks(n));
}

int main(const int argc, const char* argv[]) {
  constexpr auto n = 1920 * 1080;

  int grid_size = 64;

  if (argc > 1) {
    grid_size = std::strtol(argv[1], nullptr, 10);
  }

  std::cout << "Grid size: " << grid_size << std::endl;

  // -----------------------------------------------------------------------------

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  OneSweep one_sweep(n);
  one_sweep.attachStream(stream);

  auto mem_size = one_sweep.getMemorySize();
  // print in MB
  std::cout << "Memory size: " << mem_size / 1024 / 1024 << " MB" << std::endl;

  // std::generate_n(one_sweep.getSort(), n, [n = n]() mutable { return --n; });

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, n);
  std::generate_n(one_sweep.getSort(), n, [&dis, &gen]() { return dis(gen); });

  auto is_sorted = std::is_sorted(one_sweep.getSort(), one_sweep.getSort() + n);
  std::cout << "Is sorted: " << std::boolalpha << is_sorted << std::endl;

  Dispatch_SortKernels(one_sweep, n, grid_size, stream);

  cudaDeviceSynchronize();

  // -----------------------------------------------------------------------------

  is_sorted = std::is_sorted(one_sweep.getSort(), one_sweep.getSort() + n);
  std::cout << "Is sorted: " << std::boolalpha << is_sorted << std::endl;

  return 0;
}