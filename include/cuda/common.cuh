#pragma once

#include "constants.hpp"

__device__ __forceinline__ void InclusiveWarpScan(volatile unsigned int* t,
                                                  int index,
                                                  int strideLog) {
  if (LANE > 0) t[index] += t[index - (1 << strideLog)];
  if (LANE > 1) t[index] += t[index - (2 << strideLog)];
  if (LANE > 3) t[index] += t[index - (4 << strideLog)];
  if (LANE > 7) t[index] += t[index - (8 << strideLog)];
  if (LANE > 15) t[index] += t[index - (16 << strideLog)];
}

__device__ __forceinline__ void InclusiveWarpScanCircularShift(
    volatile unsigned int* t, int index) {
  if (LANE > 0) t[index] += t[index - 1];
  if (LANE > 1) t[index] += t[index - 2];
  if (LANE > 3) t[index] += t[index - 4];
  if (LANE > 7) t[index] += t[index - 8];
  if (LANE > 15) t[index] += t[index - 16];

  t[index] =
      __shfl_sync(__activemask(), t[index], LANE + LANE_MASK & LANE_MASK);
}

__device__ __forceinline__ void ExclusiveWarpScan(volatile unsigned int* t,
                                                  int index,
                                                  int strideLog) {
  if (LANE > 0) t[index] += t[index - (1 << strideLog)];
  if (LANE > 1) t[index] += t[index - (2 << strideLog)];
  if (LANE > 3) t[index] += t[index - (4 << strideLog)];
  if (LANE > 7) t[index] += t[index - (8 << strideLog)];
  if (LANE > 15) t[index] += t[index - (16 << strideLog)];

  t[index] = LANE ? t[index - (1 << strideLog)] : 0;
}
