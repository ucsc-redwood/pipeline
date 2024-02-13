#include <benchmark/benchmark.h>

#include "bm_01_morton.hpp"
#include "bm_02_sort.hpp"
#include "bm_03_unique.hpp"
#include "common.hpp"

BENCHMARK_REGISTER_F(MyFixture, BM_Morton32)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(MyFixture, BM_RadixSort)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(MyFixture, BM_Unique)
    ->RangeMultiplier(2)
    ->Range(1, 4)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();