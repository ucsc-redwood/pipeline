#pragma once

// For Sorting stage, we don't need the fixture.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cub/cub.cuh>
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

void DispatchSortKernels(OneSweepData<4>& one_sweep,
                         const int n,
                         const int num_blocks) {
  gpu::k_GlobalHistogram_WithLogicalBlocks<<<num_blocks, globalHistThreads>>>(
      one_sweep.u_sort,
      one_sweep.u_global_histogram,
      n,
      globalHistThreadblocks(n));

  gpu::k_DigitBinning_WithLogicalBlocks<<<num_blocks, binningThreads>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort,
      one_sweep.u_sort_alt,
      one_sweep.u_pass_histograms[0],
      one_sweep.u_index,
      n,
      0,
      binningThreadblocks(n));

  gpu::k_DigitBinning_WithLogicalBlocks<<<num_blocks, binningThreads>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort_alt,
      one_sweep.u_sort,
      one_sweep.u_pass_histograms[1],
      one_sweep.u_index,
      n,
      8,
      binningThreadblocks(n));

  gpu::k_DigitBinning_WithLogicalBlocks<<<num_blocks, binningThreads>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort,
      one_sweep.u_sort_alt,
      one_sweep.u_pass_histograms[2],
      one_sweep.u_index,
      n,
      16,
      binningThreadblocks(n));

  gpu::k_DigitBinning_WithLogicalBlocks<<<num_blocks, binningThreads>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort_alt,
      one_sweep.u_sort,
      one_sweep.u_pass_histograms[3],
      one_sweep.u_index,
      n,
      24,
      binningThreadblocks(n));
}

static void BM_OneSweepSort(bm::State& st) {
  const auto num_blocks = st.range(0);

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

  for (auto _ : st) {
    CudaEventTimer timer(st, true);

    DispatchSortKernels(one_sweep, kN, num_blocks);
  }

  (cudaFree(one_sweep.u_sort));
  (cudaFree(one_sweep.u_sort_alt));
  (cudaFree(one_sweep.u_global_histogram));
  (cudaFree(one_sweep.u_index));
  for (int i = 0; i < radixPasses; ++i) {
    (cudaFree(one_sweep.u_pass_histograms[i]));
  }
}

static void BM_CubSort(bm::State& st) {
  auto u_data = AllocateManaged<unsigned int>(kN);
  auto u_data_alt = AllocateManaged<unsigned int>(kN);
  std::generate(EXE_PAR, u_data, u_data + kN, std::rand);

  size_t temp_storage_bytes = 0;
  void* d_temp_storage = nullptr;

  cub::DeviceRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, u_data, u_data_alt, kN);

  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  for (auto _ : st) {
    CudaEventTimer timer(st, true);

    cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes, u_data, u_data_alt, kN);
  }

  Free(u_data);
  Free(u_data_alt);
  Free(d_temp_storage);
}

BENCHMARK(BM_CubSort)->UseManualTime()->Iterations(10)->Unit(bm::kMillisecond);

BENCHMARK(BM_OneSweepSort)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 10)
    ->UseManualTime()
    ->Iterations(10)
    ->Unit(bm::kMillisecond);
