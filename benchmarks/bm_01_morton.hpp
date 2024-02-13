#pragma once

#include "common.hpp"

BENCHMARK_DEFINE_F(MyFixture, BM_Morton32)(bm::State& st) {
  const auto num_threads = st.range(0);
  omp_set_num_threads(num_threads);

  for (auto _ : st) {
    k_ComputeMortonCode(u_input, u_morton_out, kN, kMin, kRange);
  }
}

BENCHMARK_REGISTER_F(MyFixture, BM_Morton32)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(benchmark::kMillisecond);
