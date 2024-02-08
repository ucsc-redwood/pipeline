#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>

#include "cuda/common.cuh"
#include "cuda/constants.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace cooperative_groups;

__global__ void k_GlobalHistogram(const unsigned int* sort,
                                  unsigned int* globalHistogram,
                                  const int size,
                                  const int logical_blocks) {
  __shared__ unsigned int s_globalHistFirst[RADIX];
  __shared__ unsigned int s_globalHistSec[RADIX];
  __shared__ unsigned int s_globalHistThird[RADIX];
  __shared__ unsigned int s_globalHistFourth[RADIX];

  for (auto my_block_id = blockIdx.x; my_block_id < logical_blocks;
       my_block_id += gridDim.x) {
    // clear
    for (int i = THREAD_ID; i < RADIX; i += G_HIST_THREADS) {
      s_globalHistFirst[i] = 0;
      s_globalHistSec[i] = 0;
      s_globalHistThird[i] = 0;
      s_globalHistFourth[i] = 0;
    }
    __syncthreads();

    // Histogram
    {
      const int partitionStart = (my_block_id * G_HIST_PART_SIZE);
      const int partitionEnd = (my_block_id == logical_blocks - 1
                                    ? size
                                    : (my_block_id + 1) * G_HIST_PART_SIZE);

      for (int i = THREAD_ID + partitionStart; i < partitionEnd;
           i += G_HIST_THREADS) {
        const unsigned int key = sort[i];
        atomicAdd(&s_globalHistFirst[key & RADIX_MASK], 1);
        atomicAdd(&s_globalHistSec[key >> SEC_RADIX & RADIX_MASK], 1);
        atomicAdd(&s_globalHistThird[key >> THIRD_RADIX & RADIX_MASK], 1);
        atomicAdd(&s_globalHistFourth[key >> FOURTH_RADIX], 1);
      }
    }
    __syncthreads();

    // exclusive prefix sum over the counts
    for (int i = THREAD_ID; i < RADIX; i += G_HIST_THREADS) {
      InclusiveWarpScanCircularShift(s_globalHistFirst, i);
      InclusiveWarpScanCircularShift(s_globalHistSec, i);
      InclusiveWarpScanCircularShift(s_globalHistThird, i);
      InclusiveWarpScanCircularShift(s_globalHistFourth, i);
    }
    __syncthreads();

    if (LANE < (RADIX >> LANE_LOG) && WARP_INDEX == 0) {
      InclusiveWarpScan(s_globalHistFirst, (LANE << LANE_LOG), LANE_LOG);
      InclusiveWarpScan(s_globalHistSec, (LANE << LANE_LOG), LANE_LOG);
      InclusiveWarpScan(s_globalHistThird, (LANE << LANE_LOG), LANE_LOG);
      InclusiveWarpScan(s_globalHistFourth, (LANE << LANE_LOG), LANE_LOG);
    }
    __syncthreads();

    // Atomically add to device memory
    {
      int i = THREAD_ID;
      atomicAdd(&globalHistogram[i],
                (LANE ? s_globalHistFirst[i] : 0) +
                    (WARP_INDEX
                         ? __shfl_sync(
                               0xffffffff, s_globalHistFirst[i - LANE_COUNT], 0)
                         : 0));
      atomicAdd(
          &globalHistogram[i + SEC_RADIX_START],
          (LANE ? s_globalHistSec[i] : 0) +
              (WARP_INDEX
                   ? __shfl_sync(0xffffffff, s_globalHistSec[i - LANE_COUNT], 0)
                   : 0));
      atomicAdd(&globalHistogram[i + THIRD_RADIX_START],
                (LANE ? s_globalHistThird[i] : 0) +
                    (WARP_INDEX
                         ? __shfl_sync(
                               0xffffffff, s_globalHistThird[i - LANE_COUNT], 0)
                         : 0));
      atomicAdd(
          &globalHistogram[i + FOURTH_RADIX_START],
          (LANE ? s_globalHistFourth[i] : 0) +
              (WARP_INDEX
                   ? __shfl_sync(
                         0xffffffff, s_globalHistFourth[i - LANE_COUNT], 0)
                   : 0));

      for (i += G_HIST_THREADS; i < RADIX; i += G_HIST_THREADS) {
        atomicAdd(
            &globalHistogram[i],
            (LANE ? s_globalHistFirst[i] : 0) +
                __shfl_sync(0xffffffff, s_globalHistFirst[i - LANE_COUNT], 0));
        atomicAdd(
            &globalHistogram[i + SEC_RADIX_START],
            (LANE ? s_globalHistSec[i] : 0) +
                __shfl_sync(0xffffffff, s_globalHistSec[i - LANE_COUNT], 0));
        atomicAdd(
            &globalHistogram[i + THIRD_RADIX_START],
            (LANE ? s_globalHistThird[i] : 0) +
                __shfl_sync(0xffffffff, s_globalHistThird[i - LANE_COUNT], 0));
        atomicAdd(
            &globalHistogram[i + FOURTH_RADIX_START],
            (LANE ? s_globalHistFourth[i] : 0) +
                __shfl_sync(0xffffffff, s_globalHistFourth[i - LANE_COUNT], 0));
      }
    }
  }  // iteration over logical blocks
}

// ============================================================================
// Original
// ============================================================================

#define G_HIST_PART_START \
  (blockIdx.x * G_HIST_PART_SIZE)  // Starting offset of a partition tile
#define G_HIST_PART_END \
  (blockIdx.x == gridDim.x - 1 ? size : (blockIdx.x + 1) * G_HIST_PART_SIZE)

__global__ void k_GlobalHistogram_Original(const unsigned int* sort,
                                           unsigned int* globalHistogram,
                                           int size) {
  __shared__ unsigned int s_globalHistFirst[RADIX];
  __shared__ unsigned int s_globalHistSec[RADIX];
  __shared__ unsigned int s_globalHistThird[RADIX];
  __shared__ unsigned int s_globalHistFourth[RADIX];

  // clear
  for (int i = THREAD_ID; i < RADIX; i += G_HIST_THREADS) {
    s_globalHistFirst[i] = 0;
    s_globalHistSec[i] = 0;
    s_globalHistThird[i] = 0;
    s_globalHistFourth[i] = 0;
  }
  __syncthreads();

  // Histogram
  {
    const auto partitionEnd = G_HIST_PART_END;
    const auto partitionStart = G_HIST_PART_START;

    // DEBUG
    // if (THREAD_ID == 0) {
    // printf("partitionStart = %d, partitionEnd = %d\n",
    //  partitionStart,
    //  partitionEnd);
    // }

    for (int i = THREAD_ID + G_HIST_PART_START; i < partitionEnd;
         i += G_HIST_THREADS) {
      const unsigned int key = sort[i];
      atomicAdd(&s_globalHistFirst[key & RADIX_MASK], 1);
      atomicAdd(&s_globalHistSec[key >> SEC_RADIX & RADIX_MASK], 1);
      atomicAdd(&s_globalHistThird[key >> THIRD_RADIX & RADIX_MASK], 1);
      atomicAdd(&s_globalHistFourth[key >> FOURTH_RADIX], 1);
    }
  }
  __syncthreads();

  // exclusive prefix sum over the counts
  for (int i = THREAD_ID; i < RADIX; i += G_HIST_THREADS) {
    InclusiveWarpScanCircularShift(s_globalHistFirst, i);
    InclusiveWarpScanCircularShift(s_globalHistSec, i);
    InclusiveWarpScanCircularShift(s_globalHistThird, i);
    InclusiveWarpScanCircularShift(s_globalHistFourth, i);
  }
  __syncthreads();

  if (LANE < (RADIX >> LANE_LOG) && WARP_INDEX == 0) {
    InclusiveWarpScan(s_globalHistFirst, (LANE << LANE_LOG), LANE_LOG);
    InclusiveWarpScan(s_globalHistSec, (LANE << LANE_LOG), LANE_LOG);
    InclusiveWarpScan(s_globalHistThird, (LANE << LANE_LOG), LANE_LOG);
    InclusiveWarpScan(s_globalHistFourth, (LANE << LANE_LOG), LANE_LOG);
  }
  __syncthreads();

  // Atomically add to device memory
  {
    int i = THREAD_ID;
    atomicAdd(
        &globalHistogram[i],
        (LANE ? s_globalHistFirst[i] : 0) +
            (WARP_INDEX
                 ? __shfl_sync(0xffffffff, s_globalHistFirst[i - LANE_COUNT], 0)
                 : 0));
    atomicAdd(
        &globalHistogram[i + SEC_RADIX_START],
        (LANE ? s_globalHistSec[i] : 0) +
            (WARP_INDEX
                 ? __shfl_sync(0xffffffff, s_globalHistSec[i - LANE_COUNT], 0)
                 : 0));
    atomicAdd(
        &globalHistogram[i + THIRD_RADIX_START],
        (LANE ? s_globalHistThird[i] : 0) +
            (WARP_INDEX
                 ? __shfl_sync(0xffffffff, s_globalHistThird[i - LANE_COUNT], 0)
                 : 0));
    atomicAdd(
        &globalHistogram[i + FOURTH_RADIX_START],
        (LANE ? s_globalHistFourth[i] : 0) +
            (WARP_INDEX ? __shfl_sync(
                              0xffffffff, s_globalHistFourth[i - LANE_COUNT], 0)
                        : 0));

    for (i += G_HIST_THREADS; i < RADIX; i += G_HIST_THREADS) {
      atomicAdd(
          &globalHistogram[i],
          (LANE ? s_globalHistFirst[i] : 0) +
              __shfl_sync(0xffffffff, s_globalHistFirst[i - LANE_COUNT], 0));
      atomicAdd(
          &globalHistogram[i + SEC_RADIX_START],
          (LANE ? s_globalHistSec[i] : 0) +
              __shfl_sync(0xffffffff, s_globalHistSec[i - LANE_COUNT], 0));
      atomicAdd(
          &globalHistogram[i + THIRD_RADIX_START],
          (LANE ? s_globalHistThird[i] : 0) +
              __shfl_sync(0xffffffff, s_globalHistThird[i - LANE_COUNT], 0));
      atomicAdd(
          &globalHistogram[i + FOURTH_RADIX_START],
          (LANE ? s_globalHistFourth[i] : 0) +
              __shfl_sync(0xffffffff, s_globalHistFourth[i - LANE_COUNT], 0));
    }
  }
}
