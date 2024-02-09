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

  auto [u_sort, u_tree] = gpu::MakeRadixTree_Fake();
  constexpr auto n_unique = kN;

  auto u_edge_count = AllocateManaged<int>(u_tree.n_nodes);
  gpu::k_EdgeCount_Kernel<<<n_blocks, block_size>>>(
      u_tree.prefixN, u_tree.parent, u_edge_count, u_tree.n_nodes);
  BENCH_CUDA_TRY(cudaDeviceSynchronize());

  auto u_count_prefix_sum = AllocateManaged<int>(u_tree.n_nodes + 1);

  [[maybe_unused]] auto _ =
      k_PartialSum(u_edge_count, 0, n_unique, u_count_prefix_sum);
  u_count_prefix_sum[0] = 0;

  const auto num_oct_nodes = u_count_prefix_sum[u_tree.n_nodes];

  auto u_oct_nodes = AllocateManaged<gpu::OctNode>(num_oct_nodes);

  const auto root_level = u_tree.prefixN[0] / 3;
  const auto root_prefix = u_sort[0] >> (morton_bits - (3 * root_level));

  cpu::morton32_to_xyz(&u_oct_nodes[0].corner,
                       root_prefix << (morton_bits - (3 * root_level)),
                       kMin,
                       kRange);
  u_oct_nodes[0].cell_size = kRange;

  for (auto _ : st) {
    cuda_event_timer timer(st, true);
    // k_MakeOctNodes(u_oct_nodes,
    //     u_count_prefix_sum,
    //     u_edge_count,
    //     u_sort,
    //     tree.prefixN,
    //     tree.parent,
    //     kMin,
    //     kRange,
    //     num_oct_nodes);

    gpu::k_MakeOctNodes<<<n_blocks, block_size>>>(u_oct_nodes,
                                                  u_count_prefix_sum,
                                                  u_edge_count,
                                                  u_sort,
                                                  u_tree.prefixN,
                                                  u_tree.parent,
                                                  kMin,
                                                  kRange,
                                                  num_oct_nodes);
  }

  Free(u_sort);
  Free(u_tree.prefixN);
  Free(u_tree.hasLeafLeft);
  Free(u_tree.hasLeafRight);
  Free(u_tree.leftChild);
  Free(u_tree.parent);
  Free(u_edge_count);
  Free(u_count_prefix_sum);
}

BENCHMARK(BM_MakeOctreeNode)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 10)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
