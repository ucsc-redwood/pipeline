#pragma once

#include "fixture.hpp"

BENCHMARK_DEFINE_F(CpuFixture, BM_EdgeCount)(bm::State& st) {
  const auto num_threads = st.range(0);
  omp_set_num_threads(num_threads);

  for (auto _ : st) {
    k_EdgeCount(tree.prefixN, tree.parent, u_edge_count_out, tree.n_nodes);
  }
}

BENCHMARK_REGISTER_F(CpuFixture, BM_EdgeCount)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(bm::kMillisecond);