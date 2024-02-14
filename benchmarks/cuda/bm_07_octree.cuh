#pragma once

#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>
#include <numeric>

namespace bm = benchmark;

#include "config.cuh"
#include "cuda/kernels/all.cuh"
#include "cuda_bench_helper.cuh"
#include "shared/morton.h"
#include "shared/types.h"

static void BM_MakeOctreeNodes(bm::State& st) {
  const auto num_blocks = st.range(0);
  const auto block_size = DetermineBlockSizeAndDisplay(gpu::k_MakeOctNodes, st);

  auto u_sort = AllocateManaged<unsigned int>(kN);
  std::iota(u_sort, u_sort + kN, 0);

  RadixTreeData tree;
  tree.n_nodes = kN - 1;
  tree.prefixN = AllocateManaged<uint8_t>(tree.n_nodes);
  tree.hasLeafLeft = AllocateManaged<bool>(tree.n_nodes);
  tree.hasLeafRight = AllocateManaged<bool>(tree.n_nodes);
  tree.leftChild = AllocateManaged<int>(tree.n_nodes);
  tree.parent = AllocateManaged<int>(tree.n_nodes);

  gpu::k_BuildRadixTree<<<64, 768>>>(kN,
                                     u_sort,
                                     tree.prefixN,
                                     tree.hasLeafLeft,
                                     tree.hasLeafRight,
                                     tree.leftChild,
                                     tree.parent);
  cudaDeviceSynchronize();

  auto u_edge_count = AllocateManaged<int>(tree.n_nodes);
  gpu::k_EdgeCount<<<num_blocks, block_size>>>(
      tree.prefixN, tree.parent, u_edge_count, tree.n_nodes);
  cudaDeviceSynchronize();

  auto u_prefix_sum = AllocateManaged<int>(tree.n_nodes + 1);
  std::partial_sum(u_edge_count, u_edge_count + tree.n_nodes, u_prefix_sum);
  u_prefix_sum[0] = 0;

  auto u_oct_nodes = AllocateManaged<OctNode>(tree.n_nodes);

  for (auto _ : st) {
    CudaEventTimer timer(st, true);

    const auto root_level = tree.prefixN[0] / 3;
    const auto root_prefix = u_sort[0] >> (morton_bits - (3 * root_level));

    shared::morton32_to_xyz(&u_oct_nodes[0].corner,
                            root_prefix << (morton_bits - (3 * root_level)),
                            kMin,
                            kRange);
    u_oct_nodes[0].cell_size = kRange;

    gpu::k_MakeOctNodes<<<num_blocks, block_size>>>(
        u_oct_nodes,
        u_prefix_sum,
        u_edge_count,
        u_sort,
        tree.prefixN,
        tree.parent,
        kMin,
        kRange,
        tree.n_nodes /* Yanwen Verified */);
  }

  Free(u_sort);
  Free(tree.prefixN);
  Free(tree.hasLeafLeft);
  Free(tree.hasLeafRight);
  Free(tree.leftChild);
  Free(tree.parent);
  Free(u_edge_count);
  Free(u_prefix_sum);
  Free(u_oct_nodes);
}

static void BM_LinkLeafNodes(bm::State& st) {
  const auto num_blocks = st.range(0);
  const auto block_size =
      DetermineBlockSizeAndDisplay(gpu::k_LinkLeafNodes, st);

  auto u_sort = AllocateManaged<unsigned int>(kN);
  std::iota(u_sort, u_sort + kN, 0);

  RadixTreeData tree;
  tree.n_nodes = kN - 1;
  tree.prefixN = AllocateManaged<uint8_t>(tree.n_nodes);
  tree.hasLeafLeft = AllocateManaged<bool>(tree.n_nodes);
  tree.hasLeafRight = AllocateManaged<bool>(tree.n_nodes);
  tree.leftChild = AllocateManaged<int>(tree.n_nodes);
  tree.parent = AllocateManaged<int>(tree.n_nodes);

  gpu::k_BuildRadixTree<<<64, 768>>>(kN,
                                     u_sort,
                                     tree.prefixN,
                                     tree.hasLeafLeft,
                                     tree.hasLeafRight,
                                     tree.leftChild,
                                     tree.parent);
  cudaDeviceSynchronize();

  auto u_edge_count = AllocateManaged<int>(tree.n_nodes);
  gpu::k_EdgeCount<<<num_blocks, block_size>>>(
      tree.prefixN, tree.parent, u_edge_count, tree.n_nodes);
  cudaDeviceSynchronize();

  auto u_prefix_sum = AllocateManaged<int>(tree.n_nodes + 1);
  std::partial_sum(u_edge_count, u_edge_count + tree.n_nodes, u_prefix_sum);
  u_prefix_sum[0] = 0;

  auto u_oct_nodes = AllocateManaged<OctNode>(tree.n_nodes);

  const auto root_level = tree.prefixN[0] / 3;
  const auto root_prefix = u_sort[0] >> (morton_bits - (3 * root_level));

  shared::morton32_to_xyz(&u_oct_nodes[0].corner,
                          root_prefix << (morton_bits - (3 * root_level)),
                          kMin,
                          kRange);
  u_oct_nodes[0].cell_size = kRange;

  gpu::k_MakeOctNodes<<<64, block_size>>>(u_oct_nodes,
                                          u_prefix_sum,
                                          u_edge_count,
                                          u_sort,
                                          tree.prefixN,
                                          tree.parent,
                                          kMin,
                                          kRange,
                                          tree.n_nodes /* Yanwen Verified */);

  for (auto _ : st) {
    CudaEventTimer timer(st, true);

    gpu::k_LinkLeafNodes<<<num_blocks, block_size>>>(u_oct_nodes,
                                                     u_prefix_sum,
                                                     u_edge_count,
                                                     u_sort,
                                                     tree.hasLeafLeft,
                                                     tree.hasLeafRight,
                                                     tree.prefixN,
                                                     tree.parent,
                                                     tree.leftChild,
                                                     tree.n_nodes);
  }

  Free(u_sort);
  Free(tree.prefixN);
  Free(tree.hasLeafLeft);
  Free(tree.hasLeafRight);
  Free(tree.leftChild);
  Free(tree.parent);
  Free(u_edge_count);
  Free(u_prefix_sum);
  Free(u_oct_nodes);
}

BENCHMARK(BM_MakeOctreeNodes)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 10)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_LinkLeafNodes)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 10)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

// #include "fixture.cuh"

// __global__ void empty_kernel() {}

// BENCHMARK_DEFINE_F(GpuFixture, BM_MakeOctreeNodes)(bm::State& st) {
//   const auto num_blocks = st.range(0);
//   const auto block_size = DetermineBlockSizeAndDisplay(gpu::k_MakeOctNodes,
//   st);

//   OctNode* u_oct_nodes = AllocateManaged<OctNode>(num_oct_nodes);

//   for (auto _ : st) {
//     CudaEventTimer timer(st, true);

//     const auto root_level = tree.prefixN[0] / 3;
//     const auto root_prefix = u_sort[0] >> (morton_bits - (3 * root_level));

//     shared::morton32_to_xyz(&u_oct_nodes[0].corner,
//                             root_prefix << (morton_bits - (3 * root_level)),
//                             kMin,
//                             kRange);
//     u_oct_nodes[0].cell_size = kRange;

//     empty_kernel<<<1, 1>>>();

//     // gpu::k_MakeOctNodes<<<num_blocks, block_size>>>(
//     //     u_oct_nodes,
//     //     u_prefix_sum,
//     //     u_edge_count,
//     //     u_sort,
//     //     tree.prefixN,
//     //     tree.parent,
//     //     kMin,
//     //     kRange,
//     //     tree.n_nodes /* Yanwen Verified */);
//   }

//   Free(u_oct_nodes);
// }

// BENCHMARK_REGISTER_F(GpuFixture, BM_MakeOctreeNodes)
//     ->RangeMultiplier(2)
//     ->Range(1, 1 << 10)
//     ->UseManualTime()
//     ->Iterations(1)
//     ->Unit(bm::kMillisecond);
