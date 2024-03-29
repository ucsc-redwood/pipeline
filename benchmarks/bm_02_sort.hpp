#pragma once

#include "fixture.hpp"

BENCHMARK_DEFINE_F(CpuFixture, BM_RadixSort)(bm::State& st) {
  const auto num_threads = st.range(0);
  omp_set_num_threads(num_threads);

  // have to use new here because this sorting will modify on the input
  auto data = new (u_morton_out) unsigned int[kN];
  k_InitDescending(data, kN);

  for (auto _ : st) {
    k_SimpleRadixSort(data, kN);
  }
}

BENCHMARK_REGISTER_F(CpuFixture, BM_RadixSort)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(bm::kMillisecond);