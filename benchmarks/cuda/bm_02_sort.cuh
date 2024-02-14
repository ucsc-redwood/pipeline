#pragma once

// For Sorting stage, we don't need the fixture.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <random>

#include "cuda_bench_helper.cuh"
#include "shared/types.h"

namespace bm = benchmark;

// TODO: merge them
const int radix = 256;
const int radixPasses = 4;
const int partitionSize = 7680;
const int globalHistPartitionSize = 65536;
const int globalHistThreads = 128;
const int binningThreads = 512;  // 2080 super seems to really like 512

constexpr int binningThreadblocks(const int size) {
  return (size + partitionSize - 1) / partitionSize;
}

constexpr int globalHistThreadblocks(const int size) {
  return (size + globalHistPartitionSize - 1) / globalHistPartitionSize;
}

void DispatchSortKernels(OneSweepData<4>& one_sweep, const int n) {
  gpu::k_GlobalHistogram<<<globalHistThreadblocks(n), globalHistThreads>>>(
      one_sweep.u_sort, one_sweep.u_global_histogram, n);

  gpu::k_DigitBinning<<<binningThreadblocks(n), binningThreads>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort,
      one_sweep.u_sort_alt,
      one_sweep.u_pass_histograms[0],
      one_sweep.u_index,
      n,
      0);

  gpu::k_DigitBinning<<<binningThreadblocks(n), binningThreads>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort_alt,
      one_sweep.u_sort,
      one_sweep.u_pass_histograms[1],
      one_sweep.u_index,
      n,
      8);

  gpu::k_DigitBinning<<<binningThreadblocks(n), binningThreads>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort,
      one_sweep.u_sort_alt,
      one_sweep.u_pass_histograms[2],
      one_sweep.u_index,
      n,
      16);

  gpu::k_DigitBinning<<<binningThreadblocks(n), binningThreads>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort_alt,
      one_sweep.u_sort,
      one_sweep.u_pass_histograms[3],
      one_sweep.u_index,
      n,
      24);
}

static void BM_OneSweepSort(bm::State& state) {
  const auto num_blocks = state.range(0);

  OneSweepData<4> one_sweep;
  one_sweep.u_sort = AllocateManaged<unsigned int>(kN);
  one_sweep.u_sort_alt = AllocateManaged<unsigned int>(kN);

  one_sweep.u_global_histogram =
      AllocateManaged<unsigned int>(radix * radixPasses);
  one_sweep.u_index = AllocateManaged<unsigned int>(radixPasses);
  for (int i = 0; i < radixPasses; ++i) {
    one_sweep.u_pass_histograms[i] =
        AllocateManaged<unsigned int>(radix * binningThreadblocks(kN));
  }

  std::generate(EXE_PAR, one_sweep.u_sort, one_sweep.u_sort + kN, std::rand);

  for (auto _ : state) {
    CudaEventTimer timer(state, true);

    DispatchSortKernels(one_sweep, kN);
  }

  (cudaFree(one_sweep.u_sort));
  (cudaFree(one_sweep.u_sort_alt));
  (cudaFree(one_sweep.u_global_histogram));
  (cudaFree(one_sweep.u_index));
  for (int i = 0; i < radixPasses; ++i) {
    (cudaFree(one_sweep.u_pass_histograms[i]));
  }
}

BENCHMARK(BM_OneSweepSort)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 10)
    ->UseManualTime()
    ->Iterations(10)
    ->Unit(bm::kMillisecond);
