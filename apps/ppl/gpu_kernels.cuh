#pragma once

#include <spdlog/spdlog.h>

#include "app_params.hpp"
#include "cuda/kernels/all.cuh"
#include "one_sweep.cuh"

/// All Gpu kerenls are wrapped with a 'grid_size' and 'stream' parameter.

namespace gpu {

void Disptach_InitRandomVec4(glm::vec4* u_data,
                             const AppParams& params,
                             // --- new parameters
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
                            // --- new parameters
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
                          // --- new parameters
                          const int grid_size,
                          const cudaStream_t& stream) {
  constexpr int globalHistThreads = 128;
  constexpr int binningThreads = 512;

  spdlog::debug(
      "Dispatching k_GlobalHistogram_WithLogicalBlocks with ({} "
      "blocks, {} threads)",
      grid_size,
      globalHistThreads);

  gpu::k_GlobalHistogram_WithLogicalBlocks<<<grid_size,
                                             globalHistThreads,
                                             0,
                                             stream>>>(
      one_sweep.u_sort,
      one_sweep.u_global_histogram,
      n,
      OneSweep::globalHistThreadblocks(n));

  spdlog::debug(
      "Dispatching k_DigitBinning_WithLogicalBlocks with ({} "
      "blocks, {} threads)",
      grid_size,
      binningThreads);

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

void Dispatch_CountUnique(unsigned int* keys,
                          int* num_unique_out,
                          int n,
                          // --- new parameters
                          const int grid_size,
                          const cudaStream_t& stream) {
  constexpr auto block_size = 768;

  spdlog::debug("Dispatching k_CountUnique with ({} blocks, {} threads)",
                grid_size,
                block_size);

  gpu::k_CountUnique<<<grid_size, block_size, 0, stream>>>(
      keys, num_unique_out, n);
}

void Dispatch_BuildRadixTree(const int* num_unique /* addr #unique */,
                             const unsigned int* codes,
                             uint8_t* prefix_n,
                             bool* has_leaf_left,
                             bool* has_leaf_right,
                             int* left_child,
                             int* parent,
                             // --- new parameters
                             const int grid_size,
                             const cudaStream_t& stream) {
  constexpr auto block_size = 768;

  spdlog::debug("Dispatching k_BuildRadixTree with ({} blocks, {} threads)",
                grid_size,
                block_size);

  gpu::k_BuildRadixTree_Deps<<<grid_size, block_size, 0, stream>>>(
      num_unique,
      codes,
      prefix_n,
      has_leaf_left,
      has_leaf_right,
      left_child,
      parent);
}

void Dispatch_EdgeCount(const uint8_t* prefix_n,
                        const int* parents,
                        int* edge_count,
                        const int* n_unique,  // changed to 'unique'
                        // --- new parameters
                        const int grid_size,
                        const cudaStream_t& stream) {
  constexpr auto block_size = 768;

  spdlog::debug("Dispatching k_EdgeCount_Deps with ({} blocks, {} threads)",
                grid_size,
                block_size);

  gpu::k_EdgeCount_Deps<<<grid_size, block_size, 0, stream>>>(
      prefix_n, parents, edge_count, n_unique);
}

void Dispatch_MakeOctree() {}

}  // namespace gpu
