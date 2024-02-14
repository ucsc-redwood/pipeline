#pragma once

#include "fixture.hpp"

BENCHMARK_DEFINE_F(CpuFixture, BM_MakeOctreeNodes)(bm::State& st) {
  const auto num_threads = st.range(0);
  omp_set_num_threads(num_threads);

  const auto root_level = tree.prefixN[0] / 3;
  const auto root_prefix = u_morton[0] >> (morton_bits - (3 * root_level));

  shared::morton32_to_xyz(&u_oct_nodes_out[0].corner,
                       root_prefix << (morton_bits - (3 * root_level)),
                       kMin,
                       kRange);
  u_oct_nodes_out[0].cell_size = kRange;

  for (auto _ : st) {
    k_MakeOctNodes(u_oct_nodes_out,
                   u_count_prefix_sum,
                   u_edge_count,
                   u_morton,
                   tree.prefixN,
                   tree.parent,
                   kMin,
                   kRange,
                   num_oct_nodes);
  }
}

BENCHMARK_DEFINE_F(CpuFixture, BM_LinkOctreeNodes)(bm::State& st) {
  const auto num_threads = st.range(0);
  omp_set_num_threads(num_threads);

  const auto root_level = tree.prefixN[0] / 3;
  const auto root_prefix = u_morton[0] >> (morton_bits - (3 * root_level));

  shared::morton32_to_xyz(&u_oct_nodes_out[0].corner,
                       root_prefix << (morton_bits - (3 * root_level)),
                       kMin,
                       kRange);
  u_oct_nodes_out[0].cell_size = kRange;

  k_MakeOctNodes(u_oct_nodes_out,
                 u_count_prefix_sum,
                 u_edge_count,
                 u_morton,
                 tree.prefixN,
                 tree.parent,
                 kMin,
                 kRange,
                 num_oct_nodes);

  for (auto _ : st) {
    k_LinkLeafNodes(u_oct_nodes_out,
                    u_count_prefix_sum,
                    u_edge_count,
                    u_morton,
                    tree.hasLeafLeft,
                    tree.hasLeafRight,
                    tree.prefixN,
                    tree.parent,
                    tree.leftChild,
                    num_oct_nodes);
  }
}

BENCHMARK_REGISTER_F(CpuFixture, BM_MakeOctreeNodes)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(bm::kMillisecond);

BENCHMARK_REGISTER_F(CpuFixture, BM_LinkOctreeNodes)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(bm::kMillisecond);