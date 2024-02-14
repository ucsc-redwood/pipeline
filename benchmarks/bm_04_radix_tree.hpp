#pragma once

#include "fixture.hpp"

BENCHMARK_DEFINE_F(CpuFixture, BM_BuildRadixTree)(bm::State& st) {
  const auto num_threads = st.range(0);
  omp_set_num_threads(num_threads);

  for (auto _ : st) {
    k_BuildRadixTree(tree_out.n_nodes,
                     u_morton,
                     tree_out.prefixN,
                     tree_out.hasLeafLeft,
                     tree_out.hasLeafRight,
                     tree_out.leftChild,
                     tree_out.parent);
  }
}

BENCHMARK_REGISTER_F(CpuFixture, BM_BuildRadixTree)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(bm::kMillisecond);
