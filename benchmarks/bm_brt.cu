#include <benchmark/benchmark.h>

#include <glm/glm.hpp>

#include "config.hpp"
#include "cuda/kernels/edge_count.cuh"
#include "cuda/kernels/init.cuh"
#include "cuda/kernels/radix_tree.cuh"
#include "cuda_bench_helper.cuh"
#include "types/brt.hpp"

namespace bm = benchmark;

static void BM_BuildRadixTree(bm::State& st) {
  const auto n_blocks = st.range(0);
  const auto block_size =
      DetermineBlockSizeAndDisplay(gpu::k_BuildRadixTree_Kernel, st);

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

  for (auto _ : st) {
    cuda_event_timer timer(st, true);
    gpu::k_BuildRadixTree_Kernel<<<n_blocks, block_size>>>(d_tree.n_nodes,
                                                           d_sort,
                                                           d_tree.prefixN,
                                                           d_tree.hasLeafLeft,
                                                           d_tree.hasLeafRight,
                                                           d_tree.leftChild,
                                                           d_tree.parent);
  }

  Free(d_sort);
  Free(d_tree.prefixN);
  Free(d_tree.hasLeafLeft);
  Free(d_tree.hasLeafRight);
  Free(d_tree.leftChild);
  Free(d_tree.parent);
}

static void BM_EdgeCount(bm::State& st) {
  const auto n_blocks = st.range(0);
  const auto block_size =
      DetermineBlockSizeAndDisplay(gpu::k_EdgeCount_Kernel, st);

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

  gpu::k_BuildRadixTree_Kernel<<<n_blocks, 768>>>(d_tree.n_nodes,
                                                  d_sort,
                                                  d_tree.prefixN,
                                                  d_tree.hasLeafLeft,
                                                  d_tree.hasLeafRight,
                                                  d_tree.leftChild,
                                                  d_tree.parent);
  BENCH_CUDA_TRY(cudaDeviceSynchronize());

  int* d_edge_count;
  BENCH_CUDA_TRY(cudaMalloc(&d_edge_count, d_tree.n_nodes * sizeof(int)));

  for (auto _ : st) {
    cuda_event_timer timer(st, true);
    gpu::k_EdgeCount_Kernel<<<n_blocks, block_size>>>(
        d_tree.prefixN, d_tree.parent, d_edge_count, d_tree.n_nodes);
  }

  Free(d_sort);
  Free(d_tree.prefixN);
  Free(d_tree.hasLeafLeft);
  Free(d_tree.hasLeafRight);
  Free(d_tree.leftChild);
  Free(d_tree.parent);
  Free(d_edge_count);
}

BENCHMARK(BM_BuildRadixTree)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 10)  // 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_EdgeCount)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 10)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
