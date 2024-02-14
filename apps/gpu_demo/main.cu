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

const int radix = 256;
const int radixPasses = 4;
const int partitionSize = 7680;
const int globalHistPartitionSize = 65536;
const int globalHistThreads = 128;
const int binningThreads = 512;  // 2080 super seems to really like 512

constexpr int binningThreadblocks(const int size) {
  return (size + partitionSize - 1) / partitionSize;
}

constexpr int globalHistThreadblocks(const int size) {
  return (size + globalHistPartitionSize - 1) / globalHistPartitionSize;
}

void InitMemory(unsigned int* u_index,
                unsigned int* globalHistogram,
                unsigned int* firstPassHistogram,
                unsigned int* secPassHistogram,
                unsigned int* thirdPassHistogram,
                unsigned int* fourthPassHistogram,
                const int n) {
  cudaMemset(u_index, 0, radixPasses * sizeof(unsigned int));
  cudaMemset(globalHistogram, 0, radix * radixPasses * sizeof(unsigned int));
  cudaMemset(firstPassHistogram,
             0,
             radix * binningThreadblocks(n) * sizeof(unsigned int));
  cudaMemset(secPassHistogram,
             0,
             radix * binningThreadblocks(n) * sizeof(unsigned int));
  cudaMemset(thirdPassHistogram,
             0,
             radix * binningThreadblocks(n) * sizeof(unsigned int));
  cudaMemset(fourthPassHistogram,
             0,
             radix * binningThreadblocks(n) * sizeof(unsigned int));
}

void DispatchSortKernels(OneSweepData<4>& one_sweep, const int n) {
  spdlog::info("dispatching radix sort... with {} blocks",
               globalHistThreadblocks(n));

  gpu::k_GlobalHistogram<<<globalHistThreadblocks(n), globalHistThreads>>>(
      one_sweep.u_sort, one_sweep.u_global_histogram, n);

  spdlog::info("dispatching k_DigitBinning... with {} blocks",
               binningThreadblocks(n));

  gpu::k_DigitBinning<<<binningThreadblocks(n), binningThreads>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort,
      one_sweep.u_sort_alt,
      one_sweep.u_pass_histograms[0],
      one_sweep.u_index,
      n,
      0);

  gpu::k_DigitBinning<<<binningThreadblocks(n), binningThreads>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort_alt,
      one_sweep.u_sort,
      one_sweep.u_pass_histograms[1],
      one_sweep.u_index,
      n,
      8);

  gpu::k_DigitBinning<<<binningThreadblocks(n), binningThreads>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort,
      one_sweep.u_sort_alt,
      one_sweep.u_pass_histograms[2],
      one_sweep.u_index,
      n,
      16);

  gpu::k_DigitBinning<<<binningThreadblocks(n), binningThreads>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort_alt,
      one_sweep.u_sort,
      one_sweep.u_pass_histograms[3],
      one_sweep.u_index,
      n,
      24);
}

int main(const int argc, const char** argv) {
  int n = 10'000'000;
  // int n = size;
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
    AllocManaged(&one_sweep.u_pass_histograms[i],
                 radix * binningThreadblocks(n));
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
  DispatchSortKernels(one_sweep, n);
  checkCudaErrors(cudaDeviceSynchronize());

  spdlog::info("Done Sorting!");

  // peek 10 elements
  for (int i = 0; i < 10; ++i) {
    spdlog::info("one_sweep.u_sort[{}] = {}", i, one_sweep.u_sort[i]);
  }

  const auto is_sorted = std::is_sorted(one_sweep.u_sort, one_sweep.u_sort + n);
  std::cout << "is_sorted = " << std::boolalpha << is_sorted << '\n';

  // ---------------------------------------------------------------------------
  // TMP

  int* num_unique_out;
  checkCudaErrors(cudaMallocManaged(&num_unique_out, sizeof(int)));

  unsigned int* u_temp_sort;
  checkCudaErrors(cudaMallocManaged(&u_temp_sort, n * sizeof(unsigned int)));
  std::copy(one_sweep.u_sort, one_sweep.u_sort + n, u_temp_sort);

  gpu::k_CountUnique<<<1, 1>>>(u_temp_sort, num_unique_out, n);
  checkCudaErrors(cudaDeviceSynchronize());

  spdlog::info("num_unique_out = {}", *num_unique_out);

  checkCudaErrors(cudaFree(num_unique_out));
  checkCudaErrors(cudaFree(u_temp_sort));

  const auto it = std::unique(one_sweep.u_sort, one_sweep.u_sort + n);
  const auto num_unique = std::distance(one_sweep.u_sort, it);

  spdlog::info("num_unique = {}", num_unique);

  assert(num_unique == *num_unique_out);

  // ---------------------------------------------------------------------------

  RadixTreeData tree;
  tree.n_nodes = num_unique - 1;
  AllocManaged(&tree.prefixN, tree.n_nodes);
  AllocManaged(&tree.hasLeafLeft, tree.n_nodes);
  AllocManaged(&tree.hasLeafRight, tree.n_nodes);
  AllocManaged(&tree.leftChild, tree.n_nodes);
  AllocManaged(&tree.parent, tree.n_nodes);

  gpu::k_BuildRadixTree<<<my_num_blocks, 768>>>(num_unique,
                                                one_sweep.u_sort,
                                                tree.prefixN,
                                                tree.hasLeafLeft,
                                                tree.hasLeafRight,
                                                tree.leftChild,
                                                tree.parent);
  checkCudaErrors(cudaDeviceSynchronize());

  // peek 10 nodes
  for (int i = 0; i < 10; ++i) {
    spdlog::trace(
        "tree.prefixN[{}] = {}, tree.hasLeafLeft[{}] = {}, "
        "tree.hasLeafRight[{}] = {}, tree.leftChild[{}] = {}, "
        "tree.parent[{}] = {}",
        i,
        tree.prefixN[i],
        i,
        tree.hasLeafLeft[i],
        i,
        tree.hasLeafRight[i],
        i,
        tree.leftChild[i],
        i,
        tree.parent[i]);
  }

  // ---------------------------------------------------------------------------
  // Cleanup

  checkCudaErrors(cudaFree(u_data));

  // Radix Sort
  checkCudaErrors(cudaFree(one_sweep.u_sort));
  checkCudaErrors(cudaFree(one_sweep.u_sort_alt));
  checkCudaErrors(cudaFree(one_sweep.u_global_histogram));
  checkCudaErrors(cudaFree(one_sweep.u_index));
  for (int i = 0; i < radixPasses; ++i) {
    checkCudaErrors(cudaFree(one_sweep.u_pass_histograms[i]));
  }

  // Radix Tree
  checkCudaErrors(cudaFree(tree.prefixN));
  checkCudaErrors(cudaFree(tree.hasLeafLeft));
  checkCudaErrors(cudaFree(tree.hasLeafRight));
  checkCudaErrors(cudaFree(tree.leftChild));
  checkCudaErrors(cudaFree(tree.parent));

  return 0;
}