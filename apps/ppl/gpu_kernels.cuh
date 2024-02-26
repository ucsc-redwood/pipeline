#pragma once

#include <spdlog/spdlog.h>

#include "app_params.hpp"
#include "cuda/kernels/all.cuh"
#include "one_sweep.cuh"
#include "shared/types.h"

/// All Gpu kerenls are wrapped with a 'grid_size' and 'stream' parameter.

namespace gpu {

void Disptach_InitRandomVec4(glm::vec4* u_data,
                             const AppParams& params,
                             const int grid_size,
                             const cudaStream_t& stream) {
  constexpr auto block_size = 768;

  spdlog::debug("Dispatching k_InitRandomVec4 with ({} blocks, {} threads)",
                grid_size,
                block_size);

  k_InitRandomVec4<<<grid_size, block_size, 0, stream>>>(
      u_data, params.n, params.min, params.range, params.seed);
}

void Dispatch_MortonCompute(const glm::vec4* u_points,
                            unsigned int* u_morton_keys,
                            const AppParams& params,
                            const int grid_size,
                            const cudaStream_t& stream) {
  constexpr auto block_size = 768;

  spdlog::debug("Dispatching k_ComputeMortonCode with ({} blocks, {} threads)",
                grid_size,
                block_size);

  k_ComputeMortonCode<<<grid_size, block_size, 0, stream>>>(
      u_points, u_morton_keys, params.n, params.min, params.range);
}

void Dispatch_SortKernels(OneSweep& one_sweep,
                          const int n,
                          const int grid_size,
                          const cudaStream_t& stream) {
  constexpr int globalHistThreads = 128;
  constexpr int binningThreads = 512;

  gpu::k_GlobalHistogram_WithLogicalBlocks<<<grid_size,
                                             globalHistThreads,
                                             0,
                                             stream>>>(
      one_sweep.u_sort,
      one_sweep.u_global_histogram,
      n,
      OneSweep::globalHistThreadblocks(n));

  gpu::k_DigitBinning_WithLogicalBlocks<<<grid_size,
                                          binningThreads,
                                          0,
                                          stream>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort,
      one_sweep.u_sort_alt,
      one_sweep.u_pass_histograms[0],
      one_sweep.u_index,
      n,
      0,
      OneSweep::binningThreadblocks(n));

  gpu::k_DigitBinning_WithLogicalBlocks<<<grid_size,
                                          binningThreads,
                                          0,
                                          stream>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort_alt,
      one_sweep.u_sort,
      one_sweep.u_pass_histograms[1],
      one_sweep.u_index,
      n,
      8,
      OneSweep::binningThreadblocks(n));

  gpu::k_DigitBinning_WithLogicalBlocks<<<grid_size,
                                          binningThreads,
                                          0,
                                          stream>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort,
      one_sweep.u_sort_alt,
      one_sweep.u_pass_histograms[2],
      one_sweep.u_index,
      n,
      16,
      OneSweep::binningThreadblocks(n));

  gpu::k_DigitBinning_WithLogicalBlocks<<<grid_size,
                                          binningThreads,
                                          0,
                                          stream>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort_alt,
      one_sweep.u_sort,
      one_sweep.u_pass_histograms[3],
      one_sweep.u_index,
      n,
      24,
      OneSweep::binningThreadblocks(n));
}

// //
// ---------------------------------------------------------------------------
// // Onesweep Sort
// //
// ---------------------------------------------------------------------------

// // namespace sort {

// // constexpr int radix = 256;
// // constexpr int radixPasses = 4;

// // }  // namespace sort

// template <int Radix, int RadixPass>
// class OneSweepHelper {
//  public:
//   static constexpr int binningThreadblocks(const int size) {
//     // Need to match to the numbers in the '.cu' file in gpu source code
//     constexpr int partitionSize = 7680;
//     return (size + partitionSize - 1) / partitionSize;
//   }

//   static constexpr int globalHistThreadblocks(const int size) {
//     // Need to match to the numbers in the '.cu' file in gpu source code
//     constexpr int globalHistPartitionSize = 65536;
//     return (size + globalHistPartitionSize - 1) / globalHistPartitionSize;
//   }

//   static OneSweepData<RadixPass> CreateOnesweepData(const int n) {
//     OneSweepData<RadixPass> one_sweep;
//     cudaMallocManaged(&one_sweep.u_sort, n * sizeof(unsigned int));
//     cudaMallocManaged(&one_sweep.u_sort_alt, n * sizeof(unsigned int));
//     cudaMallocManaged(&one_sweep.u_index, RadixPass * sizeof(unsigned int));
//     cudaMallocManaged(&one_sweep.u_global_histogram,
//                       Radix * RadixPass * sizeof(unsigned int));
//     for (int i = 0; i < RadixPass; i++) {
//       cudaMallocManaged(
//           &one_sweep.u_pass_histograms[i],
//           Radix * sort::binningThreadblocks(n) * sizeof(unsigned int));
//     }
//     return one_sweep;
//   }

//   static void AttachOnesweepStream(OneSweepData<RadixPass>& one_sweep,
//                                    const cudaStream_t& stream) {
//     cudaStreamAttachMemAsync(stream, one_sweep.u_sort, 0,
//     cudaMemAttachSingle); cudaStreamAttachMemAsync(
//         stream, one_sweep.u_sort_alt, 0, cudaMemAttachSingle);
//     cudaStreamAttachMemAsync(stream, one_sweep.u_index, 0,
//     cudaMemAttachSingle); cudaStreamAttachMemAsync(
//         stream, one_sweep.u_global_histogram, 0, cudaMemAttachSingle);
//     for (int i = 0; i < RadixPass; i++) {
//       cudaStreamAttachMemAsync(
//           stream, one_sweep.u_pass_histograms[i], 0, cudaMemAttachSingle);
//     }
//   }

//   static void DestroyOnesweepData(OneSweepData<RadixPass>& one_sweep) {
//     cudaFree(one_sweep.u_sort);
//     cudaFree(one_sweep.u_sort_alt);
//     cudaFree(one_sweep.u_index);
//     cudaFree(one_sweep.u_global_histogram);
//     for (int i = 0; i < RadixPass; i++) {
//       cudaFree(one_sweep.u_pass_histograms[i]);
//     }
//   }
// };

// static void Dispatch_SortKernels(OneSweepData<4>& one_sweep,
//                                  const int n,
//                                  const int grid_size,
//                                  const cudaStream_t& stream) {
//   constexpr int globalHistThreads = 128;
//   constexpr int binningThreads = 512;

//   gpu::k_GlobalHistogram_WithLogicalBlocks<<<grid_size,
//                                              globalHistThreads,
//                                              0,
//                                              stream>>>(
//       one_sweep.u_sort,
//       one_sweep.u_global_histogram,
//       n,
//       sort::globalHistThreadblocks(n));

//   gpu::k_DigitBinning_WithLogicalBlocks<<<grid_size,
//                                           binningThreads,
//                                           0,
//                                           stream>>>(
//       one_sweep.u_global_histogram,
//       one_sweep.u_sort,
//       one_sweep.u_sort_alt,
//       one_sweep.u_pass_histograms[0],
//       one_sweep.u_index,
//       n,
//       0,
//       sort::binningThreadblocks(n));

//   gpu::k_DigitBinning_WithLogicalBlocks<<<grid_size,
//                                           binningThreads,
//                                           0,
//                                           stream>>>(
//       one_sweep.u_global_histogram,
//       one_sweep.u_sort_alt,
//       one_sweep.u_sort,
//       one_sweep.u_pass_histograms[1],
//       one_sweep.u_index,
//       n,
//       8,
//       sort::binningThreadblocks(n));

//   gpu::k_DigitBinning_WithLogicalBlocks<<<grid_size,
//                                           binningThreads,
//                                           0,
//                                           stream>>>(
//       one_sweep.u_global_histogram,
//       one_sweep.u_sort,
//       one_sweep.u_sort_alt,
//       one_sweep.u_pass_histograms[2],
//       one_sweep.u_index,
//       n,
//       16,
//       sort::binningThreadblocks(n));

//   gpu::k_DigitBinning_WithLogicalBlocks<<<grid_size,
//                                           binningThreads,
//                                           0,
//                                           stream>>>(
//       one_sweep.u_global_histogram,
//       one_sweep.u_sort_alt,
//       one_sweep.u_sort,
//       one_sweep.u_pass_histograms[3],
//       one_sweep.u_index,
//       n,
//       24,
//       sort::binningThreadblocks(n));
// }

// ---------------------------------------------------------------------------
// End Onesweep Sort
// ---------------------------------------------------------------------------

void Dispatch_CountUnique(unsigned int* keys,
                          int* num_unique_out,
                          int n,
                          const int grid_size,
                          const cudaStream_t& stream) {
  constexpr auto block_size = 768;
  gpu::k_CountUnique<<<grid_size, block_size, 0, stream>>>(
      keys, num_unique_out, n);
}

}  // namespace gpu
