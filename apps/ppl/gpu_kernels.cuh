#pragma once

#include <spdlog/spdlog.h>

#include <utility>

#include "app_params.hpp"
#include "better_types/one_sweep.cuh"
#include "cuda/kernels/all.cuh"

/// All Gpu kerenls are wrapped with a 'grid_size' and 'stream' parameter.

namespace gpu {

template <typename Func, typename... Args>
void DispatchKernel(Func&& func,
                    const int grid_size,
                    const cudaStream_t stream,
                    Args&&... args) {
  // todo: lookup from table
  constexpr auto block_size = 768;

  spdlog::debug(
      "Dispatching kernel with ({} blocks, {} threads)", grid_size, block_size);

  std::forward<Func>(func)<<<grid_size, block_size, 0, stream>>>(
      std::forward<Args>(args)...);
}

inline void Dispatch_InitRandomVec4(glm::vec4* u_data,
                                    const AppParams& params,
                                    // --- new parameters
                                    const int grid_size,
                                    const cudaStream_t stream) {
  constexpr auto block_size = 768;

  spdlog::debug("Dispatching k_InitRandomVec4 with ({} blocks, {} threads)",
                grid_size,
                block_size);

  k_InitRandomVec4<<<grid_size, block_size, 0, stream>>>(
      u_data, params.n, params.min, params.range, params.seed);
}

inline void Dispatch_MortonCompute(const glm::vec4* u_points,
                                   unsigned int* u_morton_keys,
                                   const AppParams& params,
                                   // --- new parameters
                                   const int grid_size,
                                   const cudaStream_t stream) {
  constexpr auto block_size = 768;

  spdlog::debug("Dispatching k_ComputeMortonCode with ({} blocks, {} threads)",
                grid_size,
                block_size);

  k_ComputeMortonCode<<<grid_size, block_size, 0, stream>>>(
      u_points, u_morton_keys, params.n, params.min, params.range);
}

inline void Dispatch_SortKernels(const OneSweep& one_sweep,
                                 const int n,
                                 // --- new parameters
                                 const int grid_size,
                                 const cudaStream_t stream) {
  constexpr int globalHistThreads = 128;
  constexpr int binningThreads = 512;

  spdlog::debug(
      "Dispatching k_GlobalHistogram_WithLogicalBlocks with ({} "
      "blocks, {} threads)",
      grid_size,
      globalHistThreads);

  k_GlobalHistogram_WithLogicalBlocks<<<grid_size,
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

  k_DigitBinning_WithLogicalBlocks<<<grid_size, binningThreads, 0, stream>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort,
      one_sweep.u_sort_alt,
      one_sweep.u_pass_histograms[0],
      one_sweep.u_index,
      n,
      0,
      OneSweep::binningThreadblocks(n));

  k_DigitBinning_WithLogicalBlocks<<<grid_size, binningThreads, 0, stream>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort_alt,
      one_sweep.u_sort,
      one_sweep.u_pass_histograms[1],
      one_sweep.u_index,
      n,
      8,
      OneSweep::binningThreadblocks(n));

  k_DigitBinning_WithLogicalBlocks<<<grid_size, binningThreads, 0, stream>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort,
      one_sweep.u_sort_alt,
      one_sweep.u_pass_histograms[2],
      one_sweep.u_index,
      n,
      16,
      OneSweep::binningThreadblocks(n));

  k_DigitBinning_WithLogicalBlocks<<<grid_size, binningThreads, 0, stream>>>(
      one_sweep.u_global_histogram,
      one_sweep.u_sort_alt,
      one_sweep.u_sort,
      one_sweep.u_pass_histograms[3],
      one_sweep.u_index,
      n,
      24,
      OneSweep::binningThreadblocks(n));
}

inline void Dispatch_CountUnique(unsigned int* keys,
                                 int* num_unique_out,
                                 const int n,
                                 // --- new parameters
                                 const int grid_size,
                                 const cudaStream_t stream) {
  constexpr auto block_size = 768;

  spdlog::debug("Dispatching k_CountUnique with ({} blocks, {} threads)",
                grid_size,
                block_size);

  k_CountUnique<<<grid_size, block_size, 0, stream>>>(keys, num_unique_out, n);
}

inline void Dispatch_BuildRadixTree(const int* num_unique /* addr #unique */,
                                    const unsigned int* codes,
                                    uint8_t* prefix_n,
                                    bool* has_leaf_left,
                                    bool* has_leaf_right,
                                    int* left_child,
                                    int* parent,
                                    // --- new parameters
                                    const int grid_size,
                                    const cudaStream_t stream) {
  constexpr auto block_size = 768;

  spdlog::debug(
      "Dispatching k_BuildRadixTree_Deps with ({} blocks, {} threads)",
      grid_size,
      block_size);

  k_BuildRadixTree_Deps<<<grid_size, block_size, 0, stream>>>(num_unique,
                                                              codes,
                                                              prefix_n,
                                                              has_leaf_left,
                                                              has_leaf_right,
                                                              left_child,
                                                              parent);
}

inline void Dispatch_EdgeCount(const uint8_t* prefix_n,
                               const int* parents,
                               int* edge_count,
                               const int* n_unique,  // changed to 'unique'
                               // --- new parameters
                               const int grid_size,
                               const cudaStream_t stream) {
  constexpr auto block_size = 768;

  spdlog::debug("Dispatching k_EdgeCount_Deps with ({} blocks, {} threads)",
                grid_size,
                block_size);

  k_EdgeCount_Deps<<<grid_size, block_size, 0, stream>>>(
      prefix_n, parents, edge_count, n_unique);
}

template <typename... Args>
void Dispatch_MakeOctree(const int grid_size,
                         const cudaStream_t stream,
                         Args&&... args) {
  constexpr auto block_size = 768;

  spdlog::debug("Dispatching k_MakeOctNodes_Deps with ({} blocks, {} threads)",
                grid_size,
                block_size);

  v2::k_MakeOctNodes_Deps<<<grid_size, block_size, 0, stream>>>(
      std::forward<Args>(args)...);
}

template <typename... Args>
void Dispatch_LinkOctreeNodes(const int grid_size,
                              const cudaStream_t stream,
                              Args&&... args) {
  constexpr auto block_size = 768;

  spdlog::debug("Dispatching k_LinkLeafNodes_Deps with ({} blocks, {} threads)",
                grid_size,
                block_size);

  v2::k_LinkLeafNodes_Deps<<<grid_size, block_size, 0, stream>>>(
      std::forward<Args>(args)...);
}

}  // namespace gpu
