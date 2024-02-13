#include <benchmark/benchmark.h>

#include "bm_01_morton.hpp"
#include "common.hpp"

BENCHMARK_REGISTER_F(MyFixture, BM_Morton32)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();