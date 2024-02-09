#include <benchmark/benchmark.h>

#include <glm/glm.hpp>

#include "common.cuh"
#include "config.hpp"
#include "cuda/kernels/edge_count.cuh"
#include "cuda/kernels/octree.cuh"
#include "cuda/kernels/radix_tree.cuh"
#include "cuda_bench_helper.cuh"
#include "kernels/all.hpp"
#include "types/brt.hpp"
#include "types/oct.cuh"

//

namespace bm = benchmark;

static void BM_MakeOctreeNode(bm::State& st) {
  const auto n_blocks = st.range(0);
  const auto block_size = DetermineBlockSizeAndDisplay(gpu::k_MakeOctNodes, st);

  const auto n = kN;

  unsigned int* u_sort;
  BENCH_CUDA_TRY(cudaMallocManaged(&u_sort, n * sizeof(unsigned int)));

  gpu::k_InitAscendingSync(u_sort, n);

  const auto num_unique = n;

  const auto n_nodes = num_unique - 1;
  uint8_t* prefixN;
  bool* hasLeafLeft;
  bool* hasLeafRight;
  int* leftChild;
  int* parent;

  // clang-format off
  BENCH_CUDA_TRY(cudaMallocManaged(&prefixN, num_unique * sizeof(uint8_t)));
  BENCH_CUDA_TRY(cudaMallocManaged(&hasLeafLeft, num_unique * sizeof(bool)));
  BENCH_CUDA_TRY(cudaMallocManaged(&hasLeafRight, num_unique * sizeof(bool)));
  BENCH_CUDA_TRY(cudaMallocManaged(&leftChild, num_unique * sizeof(int)));
  BENCH_CUDA_TRY(cudaMallocManaged(&parent, num_unique * sizeof(int)));
  // clang-format on

  gpu::k_BuildRadixTree_Kernel<<<16, 768>>>(num_unique,
                                            u_sort,
                                            prefixN,
                                            hasLeafLeft,
                                            hasLeafRight,
                                            leftChild,
                                            parent);
  BENCH_CUDA_TRY(cudaDeviceSynchronize());

  for (auto i = 0; i < 10; ++i) {
    printf("\n");
    printf("prefixN[%d] = %d\n", i, prefixN[i]);
    printf("hasLeafLeft[%d] = %d\n", i, hasLeafLeft[i]);
    printf("hasLeafRight[%d] = %d\n", i, hasLeafRight[i]);
    printf("leftChild[%d] = %d\n", i, leftChild[i]);
    printf("parent[%d] = %d\n", i, parent[i]);
  }

  int* u_edge_count;
  BENCH_CUDA_TRY(cudaMallocManaged(&u_edge_count, n_nodes * sizeof(int)));

  gpu::k_EdgeCount_Kernel<<<16, 768>>>(prefixN, parent, u_edge_count, n_nodes);
  BENCH_CUDA_TRY(cudaDeviceSynchronize());

  // // peek 10
  // for (auto i = 0; i < 10; ++i) {
  //   printf("u_edge_count[%d] = %d\n", i, u_edge_count[i]);
  // }

  // Partial sum

  int* u_count_prefix_sum;
  BENCH_CUDA_TRY(
      cudaMallocManaged(&u_count_prefix_sum, num_unique * sizeof(int)));

  [[maybe_unused]] auto _ =
      k_PartialSum(u_edge_count, 0, num_unique, u_count_prefix_sum);
  u_count_prefix_sum[0] = 0;

  const auto num_oct_nodes = u_count_prefix_sum[n_nodes];
  st.counters["num_oct_nodes"] = num_oct_nodes;

  u_count_prefix_sum[0] = 0;

  // // peek at the first 10 prefix sums
  // for (auto i = 0; i < 10; ++i) {
  //   spdlog::info("u_count_prefix_sum[{}] = {}", i, u_count_prefix_sum[i]);
  // }

  // ---------------------------------------------------------------------------

  gpu::OctNode* u_oct_nodes;
  BENCH_CUDA_TRY(
      cudaMallocManaged(&u_oct_nodes, num_oct_nodes * sizeof(gpu::OctNode)));

  // constexpr auto morton_bits = 30;

  const auto root_level = prefixN[0] / 3;
  const auto root_prefix = u_sort[0] >> (morton_bits - (3 * root_level));

  cpu::morton32_to_xyz(&u_oct_nodes[0].corner,
                       root_prefix << (morton_bits - (3 * root_level)),
                       1.0f,
                       1024.0f);
  u_oct_nodes[0].cell_size = 1024.0f;

  for (auto _ : st) {
    cuda_event_timer timer(st, true);
    gpu::k_MakeOctNodes<<<n_blocks, block_size>>>(u_oct_nodes,
                                                  u_count_prefix_sum,
                                                  u_edge_count,
                                                  u_sort,
                                                  prefixN,
                                                  parent,
                                                  1.0f,
                                                  1024.0f,
                                                  num_oct_nodes);
  }

  Free(u_sort);
  Free(prefixN);
  Free(hasLeafLeft);
  Free(hasLeafRight);
  Free(leftChild);
  Free(parent);
  Free(u_edge_count);
  Free(u_count_prefix_sum);
  Free(u_oct_nodes);
}

// BENCHMARK(BM_MakeOctreeNode)
//     ->RangeMultiplier(2)
//     ->Range(1, 1 << 10)
//     ->UseManualTime()
//     ->Unit(bm::kMillisecond)
//     ->Iterations(1);

BENCHMARK_MAIN();
