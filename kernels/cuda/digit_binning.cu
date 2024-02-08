#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>

#include "cuda/common.cuh"
#include "cuda/constants.hpp"

using namespace cooperative_groups;

// for the chained scan with decoupled lookback
// Flag value inidicating neither inclusive sum, nor reduction of a
// partition tile is ready
constexpr auto FLAG_NOT_READY = 0;
// Flag value indicating reduction of a partition tile is ready
constexpr auto FLAG_REDUCTION = 1;
// Flag value indicating inclusive sum of a partition tile is ready
constexpr auto FLAG_INCLUSIVE = 2;
// Mask used to retrieve flag values
constexpr auto FLAG_MASK = 3;

// #define YANWEN_NUM_BLOCKS gridDim.x

// __global__ void k_DigitBinning(unsigned int* globalHistogram,
//                                unsigned int* sort,
//                                unsigned int* alt,
//                                volatile unsigned int* passHistogram,
//                                unsigned int* index,
//                                int size,
//                                unsigned int radixShift,
//                                int logical_blocks) {
//   __shared__ unsigned int s_warpHistograms[BIN_PART_SIZE];
//   __shared__ unsigned int s_localHistogram[RADIX];

//   unsigned int* s_warpHist = &s_warpHistograms[WARP_INDEX << RADIX_LOG];

//   // atomically assign partition tiles
//   if (LANE == 0 && WARP_INDEX == 0)
//     s_localHistogram[0] = atomicAdd(&index[radixShift >> 3], 1);
//   __syncthreads();
//   int partitionIndex = s_localHistogram[0];
//   __syncthreads();

//   // load global histogram into shared memory
//   if (THREAD_ID < RADIX)
//     s_localHistogram[THREAD_ID] =
//         globalHistogram[THREAD_ID + (radixShift << 5)];

// // clear
// #pragma unroll
//   for (int i = LANE; i < RADIX; i += LANE_COUNT) s_warpHist[i] = 0;

//   // load keys
//   unsigned int keys[BIN_KEYS_PER_THREAD];
// #pragma unroll
//   for (int i = 0, t = LANE + BIN_SUB_PART_START + BIN_PART_START;
//        i < BIN_KEYS_PER_THREAD;
//        ++i, t += LANE_COUNT)
//     keys[i] = sort[t];

//   // WLMS
//   unsigned int
//       _offsets[(BIN_KEYS_PER_THREAD >> 1) + (BIN_KEYS_PER_THREAD & 1 ? 1 :
//       0)];
//   unsigned short* offsets = reinterpret_cast<unsigned short*>(_offsets);
//   for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
//     unsigned int warpFlags = 0xffffffff;
//     for (int k = radixShift; k < radixShift + RADIX_LOG; ++k) {
//       const bool t2 = keys[i] >> k & 1;
//       warpFlags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
//     }

//     const unsigned int bits = __popc(warpFlags << LANE_MASK - LANE);
//     offsets[i] = s_warpHist[keys[i] >> radixShift & RADIX_MASK] + bits - 1;
//     if (bits == 1)
//       s_warpHist[keys[i] >> radixShift & RADIX_MASK] += __popc(warpFlags);
//   }
//   __syncthreads();

//   // exclusive prefix sum across the warp histograms
//   if (THREAD_ID < RADIX) {
//     const unsigned int t = THREAD_ID;
//     for (int i = t + RADIX; i < BIN_HISTS_SIZE; i += RADIX) {
//       s_warpHistograms[t] += s_warpHistograms[i];
//       s_warpHistograms[i] = s_warpHistograms[t] - s_warpHistograms[i];
//     }

//     if (partitionIndex == 0)
//       atomicAdd((unsigned int*)&passHistogram[THREAD_ID * YANWEN_NUM_BLOCKS +
//                                               partitionIndex],
//                 FLAG_INCLUSIVE | s_warpHistograms[THREAD_ID] << 2);
//     else
//       atomicAdd((unsigned int*)&passHistogram[THREAD_ID * YANWEN_NUM_BLOCKS +
//                                               partitionIndex],
//                 FLAG_REDUCTION | s_warpHistograms[THREAD_ID] << 2);
//   }
//   __syncthreads();

//   // exlusive prefix sum across the reductions
//   if (THREAD_ID < RADIX)
//     InclusiveWarpScanCircularShift(s_warpHistograms, THREAD_ID);
//   __syncthreads();

//   if (LANE < (RADIX >> LANE_LOG) && WARP_INDEX == 0)
//     ExclusiveWarpScan(s_warpHistograms, LANE << LANE_LOG, LANE_LOG);
//   __syncthreads();

//   if (THREAD_ID < RADIX && LANE)
//     s_warpHistograms[THREAD_ID] +=
//         __shfl_sync(0xfffffffe, s_warpHistograms[THREAD_ID - 1], 1);
//   __syncthreads();

//   // update offsets
//   if (WARP_INDEX) {
// #pragma unroll
//     for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
//       const unsigned int t2 = keys[i] >> radixShift & RADIX_MASK;
//       offsets[i] += s_warpHist[t2] + s_warpHistograms[t2];
//     }
//   } else {
// #pragma unroll
//     for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
//       offsets[i] += s_warpHistograms[keys[i] >> radixShift & RADIX_MASK];
//   }
//   __syncthreads();

//   // split the warps into single thread cooperative groups and lookback
//   if (partitionIndex) {
//     thread_block_tile<1> threadID = tiled_partition<1>(this_thread_block());

//     for (int i = threadID.meta_group_rank(); i < RADIX; i += BIN_THREADS) {
//       unsigned int reduction = 0;
//       for (int k = partitionIndex - 1; 0 <= k;) {
//         const unsigned int flagPayload =
//             passHistogram[i * YANWEN_NUM_BLOCKS + k];

//         if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
//           reduction += flagPayload >> 2;
//           atomicAdd((unsigned int*)&passHistogram[i * YANWEN_NUM_BLOCKS +
//                                                   partitionIndex],
//                     1 | (reduction << 2));
//           s_localHistogram[i] += reduction - s_warpHistograms[i];
//           break;
//         }

//         if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION) {
//           reduction += flagPayload >> 2;
//           k--;
//         }
//       }
//     }
//   } else {
//     if (THREAD_ID < RADIX)
//       s_localHistogram[THREAD_ID] -= s_warpHistograms[THREAD_ID];
//   }
//   __syncthreads();

// // scatter keys into shared memory
// #pragma unroll
//   for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
//     s_warpHistograms[offsets[i]] = keys[i];
//   __syncthreads();

//   // scatter runs of keys into device memory
//   for (int i = THREAD_ID; i < BIN_PART_SIZE; i += BIN_THREADS)
//     alt[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] + i]
//     =
//         s_warpHistograms[i];

//   // To handle input sizes not perfect multiples of the partition tile size
//   if (partitionIndex == YANWEN_NUM_BLOCKS - 1) {
//     __syncthreads();
//     {
//       const int tid = THREAD_ID;
//       if (tid < RADIX)
//         s_localHistogram[tid] =
//             (passHistogram[tid * YANWEN_NUM_BLOCKS + partitionIndex] >> 2) +
//             globalHistogram[tid + (radixShift << 5)];
//     }
//     __syncthreads();

//     partitionIndex++;
//     for (int i = THREAD_ID + BIN_PART_START; i < size; i += BIN_THREADS) {
//       const unsigned int key = sort[i];
//       unsigned int offset = 0xffffffff;

//       for (int k = radixShift; k < radixShift + RADIX_LOG; ++k) {
//         const bool t = key >> k & 1;
//         offset &= (t ? 0 : 0xffffffff) ^ __ballot_sync(__activemask(), t);
//       }

// #pragma unroll
//       for (int k = 0; k < BIN_WARPS; ++k) {
//         if (WARP_INDEX == k) {
//           const unsigned int t =
//               s_localHistogram[key >> radixShift & RADIX_MASK];
//           const unsigned int bits = __popc(offset << LANE_MASK - LANE);
//           if (bits == 1)
//             s_localHistogram[key >> radixShift & RADIX_MASK] +=
//             __popc(offset);
//           offset = t + bits - 1;
//         }
//         __syncthreads();
//       }

//       alt[offset] = key;
//     }
//   }
// }

// //
// ============================================================================
// Original
// ============================================================================

#define BIN_SUB_PART_START \
  (WARP_INDEX * BIN_SUB_PART_SIZE)  // Starting offset of a subpartition tile
#define BIN_PART_START \
  (partitionIndex * BIN_PART_SIZE)  // Starting offset of a partition tile

__global__ void k_DigitBinning_Original(unsigned int* globalHistogram,
                                        unsigned int* sort,
                                        unsigned int* alt,
                                        volatile unsigned int* passHistogram,
                                        unsigned int* index,
                                        int size,
                                        unsigned int radixShift) {
  __shared__ unsigned int s_warpHistograms[BIN_PART_SIZE];
  __shared__ unsigned int s_localHistogram[RADIX];

  unsigned int* s_warpHist = &s_warpHistograms[WARP_INDEX << RADIX_LOG];

  // atomically assign partition tiles
  if (LANE == 0 && WARP_INDEX == 0)
    s_localHistogram[0] = atomicAdd(&index[radixShift >> 3], 1);
  __syncthreads();
  int partitionIndex = s_localHistogram[0];
  __syncthreads();

  // load global histogram into shared memory
  if (THREAD_ID < RADIX)
    s_localHistogram[THREAD_ID] =
        globalHistogram[THREAD_ID + (radixShift << 5)];

// clear
#pragma unroll
  for (int i = LANE; i < RADIX; i += LANE_COUNT) s_warpHist[i] = 0;

  // load keys
  unsigned int keys[BIN_KEYS_PER_THREAD];
#pragma unroll
  for (int i = 0, t = LANE + BIN_SUB_PART_START + BIN_PART_START;
       i < BIN_KEYS_PER_THREAD;
       ++i, t += LANE_COUNT)
    keys[i] = sort[t];

  // WLMS
  unsigned int
      _offsets[(BIN_KEYS_PER_THREAD >> 1) + (BIN_KEYS_PER_THREAD & 1 ? 1 : 0)];
  unsigned short* offsets = reinterpret_cast<unsigned short*>(_offsets);
  for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
    unsigned int warpFlags = 0xffffffff;
    for (int k = radixShift; k < radixShift + RADIX_LOG; ++k) {
      const bool t2 = keys[i] >> k & 1;
      warpFlags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
    }

    const unsigned int bits = __popc(warpFlags << LANE_MASK - LANE);
    offsets[i] = s_warpHist[keys[i] >> radixShift & RADIX_MASK] + bits - 1;
    if (bits == 1)
      s_warpHist[keys[i] >> radixShift & RADIX_MASK] += __popc(warpFlags);
  }
  __syncthreads();

  // exclusive prefix sum across the warp histograms
  if (THREAD_ID < RADIX) {
    const unsigned int t = THREAD_ID;
    for (int i = t + RADIX; i < BIN_HISTS_SIZE; i += RADIX) {
      s_warpHistograms[t] += s_warpHistograms[i];
      s_warpHistograms[i] = s_warpHistograms[t] - s_warpHistograms[i];
    }

    if (partitionIndex == 0)
      atomicAdd(
          (unsigned int*)&passHistogram[THREAD_ID * gridDim.x + partitionIndex],
          FLAG_INCLUSIVE | s_warpHistograms[THREAD_ID] << 2);
    else
      atomicAdd(
          (unsigned int*)&passHistogram[THREAD_ID * gridDim.x + partitionIndex],
          FLAG_REDUCTION | s_warpHistograms[THREAD_ID] << 2);
  }
  __syncthreads();

  // exlusive prefix sum across the reductions
  if (THREAD_ID < RADIX)
    InclusiveWarpScanCircularShift(s_warpHistograms, THREAD_ID);
  __syncthreads();

  if (LANE < (RADIX >> LANE_LOG) && WARP_INDEX == 0)
    ExclusiveWarpScan(s_warpHistograms, LANE << LANE_LOG, LANE_LOG);
  __syncthreads();

  if (THREAD_ID < RADIX && LANE)
    s_warpHistograms[THREAD_ID] +=
        __shfl_sync(0xfffffffe, s_warpHistograms[THREAD_ID - 1], 1);
  __syncthreads();

  // update offsets
  if (WARP_INDEX) {
#pragma unroll
    for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
      const unsigned int t2 = keys[i] >> radixShift & RADIX_MASK;
      offsets[i] += s_warpHist[t2] + s_warpHistograms[t2];
    }
  } else {
#pragma unroll
    for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
      offsets[i] += s_warpHistograms[keys[i] >> radixShift & RADIX_MASK];
  }
  __syncthreads();

  // split the warps into single thread cooperative groups and lookback
  if (partitionIndex) {
    thread_block_tile<1> threadID = tiled_partition<1>(this_thread_block());

    for (int i = threadID.meta_group_rank(); i < RADIX; i += BIN_THREADS) {
      unsigned int reduction = 0;
      for (int k = partitionIndex - 1; 0 <= k;) {
        const unsigned int flagPayload = passHistogram[i * gridDim.x + k];

        if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
          reduction += flagPayload >> 2;
          atomicAdd(
              (unsigned int*)&passHistogram[i * gridDim.x + partitionIndex],
              1 | (reduction << 2));
          s_localHistogram[i] += reduction - s_warpHistograms[i];
          break;
        }

        if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION) {
          reduction += flagPayload >> 2;
          k--;
        }
      }
    }
  } else {
    if (THREAD_ID < RADIX)
      s_localHistogram[THREAD_ID] -= s_warpHistograms[THREAD_ID];
  }
  __syncthreads();

// scatter keys into shared memory
#pragma unroll
  for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
    s_warpHistograms[offsets[i]] = keys[i];
  __syncthreads();

  // scatter runs of keys into device memory
  for (int i = THREAD_ID; i < BIN_PART_SIZE; i += BIN_THREADS)
    alt[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] + i] =
        s_warpHistograms[i];

  // To handle input sizes not perfect multiples of the partition tile size
  if (partitionIndex == gridDim.x - 1) {
    __syncthreads();
    {
      const int tid = THREAD_ID;
      if (tid < RADIX)
        s_localHistogram[tid] =
            (passHistogram[tid * gridDim.x + partitionIndex] >> 2) +
            globalHistogram[tid + (radixShift << 5)];
    }
    __syncthreads();

    partitionIndex++;
    for (int i = THREAD_ID + BIN_PART_START; i < size; i += BIN_THREADS) {
      const unsigned int key = sort[i];
      unsigned int offset = 0xffffffff;

      for (int k = radixShift; k < radixShift + RADIX_LOG; ++k) {
        const bool t = key >> k & 1;
        offset &= (t ? 0 : 0xffffffff) ^ __ballot_sync(__activemask(), t);
      }

#pragma unroll
      for (int k = 0; k < BIN_WARPS; ++k) {
        if (WARP_INDEX == k) {
          const unsigned int t =
              s_localHistogram[key >> radixShift & RADIX_MASK];
          const unsigned int bits = __popc(offset << LANE_MASK - LANE);
          if (bits == 1)
            s_localHistogram[key >> radixShift & RADIX_MASK] += __popc(offset);
          offset = t + bits - 1;
        }
        __syncthreads();
      }

      alt[offset] = key;
    }
  }
}