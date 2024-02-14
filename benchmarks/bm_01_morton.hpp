#pragma once

#include "fixture.hpp"

BENCHMARK_DEFINE_F(CpuFixture, BM_Morton)(bm::State& st) {
  const auto num_threads = st.range(0);
  omp_set_num_threads(num_threads);

  for (auto _ : st) {
    k_ComputeMortonCode(u_input, u_morton_out, kN, kMin, kRange);
  }
}

BENCHMARK_REGISTER_F(CpuFixture, BM_Morton)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(bm::kMillisecond);
