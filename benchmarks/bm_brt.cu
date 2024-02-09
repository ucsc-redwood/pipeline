#include <benchmark/benchmark.h>

#include <glm/glm.hpp>

#include "config.hpp"
#include "cuda/common.cuh"
#include "cuda/kernels/radix_tree.cuh"
#include "cuda_bench_helper.cuh"
#include "kernels/init.hpp"
#include "types/brt.hpp"

namespace bm = benchmark;

static void BM_BuildRadixTree(bm::State& st) {
  const auto n_blocks = st.range(0);

  auto d_sort = AllocateDevice<unsigned int>(kN);

  gpu::k_InitAscendingSync(d_sort, kN);

  const auto num_unique = kN;

  RadixTreeData d_tree;
  d_tree.n_nodes = num_unique - 1;
  // clang-format off
  BENCH_CUDA_TRY(cudaMalloc(&d_tree.prefixN, num_unique * sizeof(uint8_t)));
  BENCH_CUDA_TRY(cudaMalloc(&d_tree.hasLeafLeft, num_unique * sizeof(bool)));
  BENCH_CUDA_TRY(cudaMalloc(&d_tree.hasLeafRight, num_unique * sizeof(bool)));
  BENCH_CUDA_TRY(cudaMalloc(&d_tree.leftChild, num_unique * sizeof(int)));
  BENCH_CUDA_TRY(cudaMalloc(&d_tree.parent, num_unique * sizeof(int)));
  // clang-format on
  const auto block_size = DetermineBlockSize(gpu::k_BuildRadixTree_Kernel);
  st.counters["block_size"] = block_size;
  // constexpr auto block_size = 768;

  for (auto _ : st) {
    cuda_event_timer timer(st, true);
    gpu::k_BuildRadixTree_Kernel<<<n_blocks, block_size>>>(num_unique,
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

BENCHMARK(BM_BuildRadixTree)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 10)  // 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
