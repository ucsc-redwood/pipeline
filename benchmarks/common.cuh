#pragma once

#include <stdexcept>

#include "config.hpp"
#include "cuda/common.cuh"
#include "cuda/kernels/init.cuh"
#include "cuda/kernels/radix_tree.cuh"
#include "types/brt.hpp"

// #define BENCH_CUDA_TRY(call)                            \
//   do {                                                  \
//     auto const status = (call);                         \
//     if (cudaSuccess != status) {                        \
//       throw std::runtime_error("CUDA error detected."); \
//     }                                                   \
//   } while (0);

// template <typename T>
// [[nodiscard]] static T *AllocateDevice(const int n) {
//   T *d_data;
//   BENCH_CUDA_TRY(cudaMalloc(&d_data, n * sizeof(T)));
//   return d_data;
// }

// template <typename T>
// [[nodiscard]] static T *AllocateHost(const int n) {
//   T *h_data;
//   BENCH_CUDA_TRY(cudaMallocHost(&h_data, n * sizeof(T)));
//   return h_data;
// }

// template <typename T>
// [[nodiscard]] static T *AllocateManaged(const int n) {
//   T *u_data;
//   BENCH_CUDA_TRY(cudaMallocManaged(&u_data, n * sizeof(T)));
//   return u_data;
// }
//
// static void Free(void *ptr) { BENCH_CUDA_TRY(cudaFree(ptr)); }

namespace gpu {

static std::pair<unsigned int *, RadixTreeData> MakeRadixTree_Fake() {
  unsigned int *u_sort;
  cudaMallocManaged(&u_sort, kN * sizeof(unsigned int));

  gpu::k_InitAscendingSync(u_sort, kN);

  constexpr auto num_unique = kN;
  RadixTreeData u_tree;
  u_tree.n_nodes = num_unique - 1;
  cudaMallocManaged(&u_tree.prefixN, num_unique * sizeof(uint8_t));
  cudaMallocManaged(&u_tree.hasLeafLeft, num_unique * sizeof(bool));
  cudaMallocManaged(&u_tree.hasLeafRight, num_unique * sizeof(bool));
  cudaMallocManaged(&u_tree.leftChild, num_unique * sizeof(int));
  cudaMallocManaged(&u_tree.parent, num_unique * sizeof(int));

  constexpr auto block_size = 768;
  constexpr auto grid_size = (num_unique + block_size - 1) / block_size;

  gpu::k_BuildRadixTree_Kernel<<<grid_size, block_size>>>(u_tree.n_nodes,
                                                          u_sort,
                                                          u_tree.prefixN,
                                                          u_tree.hasLeafLeft,
                                                          u_tree.hasLeafRight,
                                                          u_tree.leftChild,
                                                          u_tree.parent);
  cudaDeviceSynchronize();

  return {u_sort, u_tree};
}

}  // namespace gpu