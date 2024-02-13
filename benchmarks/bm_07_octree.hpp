#pragma once

#include "common.hpp"

BENCHMARK_DEFINE_F(MyFixture, BM_MakeOctreeNodes)(bm::State& st) {
  const auto num_threads = st.range(0);
  omp_set_num_threads(num_threads);

  const auto root_level = tree.prefixN[0] / 3;
  const auto root_prefix = u_morton[0] >> (morton_bits - (3 * root_level));

  cpu::morton32_to_xyz(&u_oct_nodes_out[0].corner,
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

BENCHMARK_REGISTER_F(MyFixture, BM_MakeOctreeNodes)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(benchmark::kMillisecond);
