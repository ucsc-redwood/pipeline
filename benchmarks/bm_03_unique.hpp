#pragma once

#include "common.hpp"

BENCHMARK_DEFINE_F(MyFixture, BM_Unique)(bm::State& st) {
  const auto num_threads = st.range(0);
  omp_set_num_threads(num_threads);

  for (auto _ : st) {
    const auto ret = k_CountUnique(u_morton_out, kN);
    bm::DoNotOptimize(ret);
  }
}