#pragma once

#include "fixture.hpp"

BENCHMARK_DEFINE_F(CpuFixture, BM_PartialSum)(bm::State& st) {
  const auto num_threads = st.range(0);
  omp_set_num_threads(num_threads);

  for (auto _ : st) {
    auto ret = k_PartialSum(u_edge_count, 0, n_unique, u_count_prefix_sum_out);
    u_count_prefix_sum_out[0] = 0;
    bm::DoNotOptimize(ret);
  }
}

BENCHMARK_REGISTER_F(CpuFixture, BM_PartialSum)
    ->RangeMultiplier(2)
    ->Range(1, 4)
    ->Unit(bm::kMillisecond);

