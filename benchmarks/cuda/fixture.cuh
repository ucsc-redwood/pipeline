#pragma once

#include <benchmark/benchmark.h>
// #include <omp.h>

#include <glm/glm.hpp>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>

namespace bm = benchmark;

// #include <cuda_runtime_api.h>

#include "config.cuh"
#include "cuda/kernels/all.cuh"
#include "cuda_bench_helper.cuh"
#include "shared/morton.h"
#include "shared/types.h"

// // Check if <execution> header is available
// #ifdef __has_include
// #if __has_include(<execution>)
// #include <execution>
// #define HAS_EXECUTION_HEADER 1
// #endif
// #endif

// #ifdef HAS_EXECUTION_HEADER
// #define EXE_PAR std::execution::par_unseq
// #else
// #define EXE_PAR
// #endif

class GpuFixture : public bm::Fixture {
 public:
  GpuFixture() {
#ifdef HAS_EXECUTION_HEADER
    std::cout << "*** Execution header is available. ***" << std::endl;
#else
    std::cout << "*** Execution header is not available. ***" << std::endl;
#endif
    // ----------------------------------------
    // Let's check the optimal number of threads per block first, and then
    // record it to the table.
    //

    // {
    //   // Set cuda device
    //   int device = 0;
    //   BENCH_CUDA_TRY(cudaSetDevice(device));

    //   int blockSize = 1;
    //   int minGridSize = 1;
    //   BENCH_CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(
    //       &minGridSize, &blockSize, gpu::k_ComputeMortonCode));
    //   optimal_block_size["rand"] = blockSize;
    // }

    // clang-format off
    // optimal_block_size["rand"] = DetermineBlockSize(gpu::k_InitRandomVec4);
    // optimal_block_size["morton"] = DetermineBlockSize(gpu::k_ComputeMortonCode);
    // optimal_block_size["histogram"] = DetermineBlockSize(gpu::k_GlobalHistogram);
    // optimal_block_size["binning"] = DetermineBlockSize(gpu::k_DigitBinning);
    // optimal_block_size["unique"] = DetermineBlockSize(gpu::k_CountUnique);
    // clang-format on

    // // Print the optimal block size
    // for (const auto& [name, size] : optimal_block_size) {
    //   std::cout << "Optimal block size for " << name << " is " << size
    //             << std::endl;
    // }

    // ----------------------------------------
    // Allocate memory for Fixture
    //
    u_points = AllocateManaged<glm::vec4>(kN);
    u_points_out = AllocateManaged<unsigned int>(kN);

    u_sort = AllocateManaged<unsigned int>(kN);
    u_sort_alt = AllocateManaged<unsigned int>(kN);
    u_sort_alt_out = AllocateManaged<unsigned int>(kN);

    // generate random points. (process) -> 'u_points'
    std::mt19937 gen(kRandomSeed);
    std::uniform_real_distribution<float> dis(kMin, kMax);
    std::generate(u_points, u_points + kN, [&]() {
      return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
    });

    // Do morton. 'u_points' -> (process) -> 'u_sort'
    std::transform(u_points, u_points + kN, u_sort, [](const auto& p) {
      return shared::xyz_to_morton32(p, kMin, kMax);
    });

    // {
    //   // Set cuda device
    //   int device = 0;
    //   BENCH_CUDA_TRY(cudaSetDevice(device));

    //   int blockSize = 1;
    //   int minGridSize = 1;
    //   BENCH_CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(
    //       &minGridSize, &blockSize, gpu::k_ComputeMortonCode));
    //   optimal_block_size["rand"] = blockSize;
    // }

    // std::sort()
  }

  ~GpuFixture() {
    Free(u_points);
    Free(u_points_out);
    Free(u_sort);
    Free(u_sort_alt);
    Free(u_sort_alt_out);
  }

  std::unordered_map<std::string, int> optimal_block_size;

  // stage 1 Morton
  glm::vec4* u_points;
  unsigned int* u_points_out;

  // stage 2 sort
  unsigned int* u_sort;
  unsigned int* u_sort_alt;  // no out?

  unsigned int* u_sort_alt_out;  // out?
  // stage 3

  RadixTreeData u_tree;
  RadixTreeData u_tree_out;
};
