#pragma once

#include <stdexcept>

#include "config.hpp"
#include "cuda/common.cuh"
#include "cuda/kernels/init.cuh"
#include "cuda/kernels/radix_tree.cuh"
#include "types/brt.hpp"

#define BENCH_CUDA_TRY(call)                            \
  do {                                                  \
    auto const status = (call);                         \
    if (cudaSuccess != status) {                        \
      throw std::runtime_error("CUDA error detected."); \
    }                                                   \
  } while (0);

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

namespace gpu {

static std::pair<unsigned int *, RadixTreeData> MakeRadixTree_Fake() {
  const auto d_sort = AllocateDevice<unsigned int>(kN);
  gpu::k_InitAscendingSync(d_sort, kN);

  constexpr auto num_unique = kN;
  RadixTreeData d_tree;
  d_tree.n_nodes = num_unique - 1;
  // clang-format off
    BENCH_CUDA_TRY(cudaMalloc(&d_tree.prefixN, d_tree.n_nodes * sizeof(uint8_t)));
    BENCH_CUDA_TRY(cudaMalloc(&d_tree.hasLeafLeft, d_tree.n_nodes * sizeof(bool)));
    BENCH_CUDA_TRY(cudaMalloc(&d_tree.hasLeafRight, d_tree.n_nodes * sizeof(bool)));
    BENCH_CUDA_TRY(cudaMalloc(&d_tree.leftChild, d_tree.n_nodes * sizeof(int)));
    BENCH_CUDA_TRY(cudaMalloc(&d_tree.parent, d_tree.n_nodes * sizeof(int)));
  // clang-format on

  constexpr auto block_size = 768;
  constexpr auto grid_size = (num_unique + block_size - 1) / block_size;

  gpu::k_BuildRadixTree_Kernel<<<grid_size, block_size>>>(d_tree.n_nodes,
                                                          d_sort,
                                                          d_tree.prefixN,
                                                          d_tree.hasLeafLeft,
                                                          d_tree.hasLeafRight,
                                                          d_tree.leftChild,
                                                          d_tree.parent);
  BENCH_CUDA_TRY(cudaDeviceSynchronize());

  return {d_sort, d_tree};
}

}  // namespace gpu