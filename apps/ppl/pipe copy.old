#pragma once

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <glm/glm.hpp>

#include "cuda/common/helper_cuda.hpp"
#include "shared/types.h"

struct Pipe {
  static constexpr auto kRadix = 256;
  static constexpr auto kRadixPasses = 4;

  template <typename T>
  static T* Alloc(size_t n) {
    T* ptr;
    checkCudaErrors(cudaMallocManaged(&ptr, n * sizeof(T)));
    return ptr;
  }

  // For Octree Nodes,
  // we need to allocate large enough memory, give a rough estimate
  Pipe(const int n) : n(n) {
    u_points = Alloc<glm::vec4>(n);
    one_sweep.u_sort = Alloc<unsigned int>(n);
    one_sweep.u_sort_alt = Alloc<unsigned int>(n);
    one_sweep.u_global_histogram = Alloc<unsigned int>(kRadix * kRadixPasses);
    one_sweep.u_index = Alloc<unsigned int>(kRadixPasses);
    for (int i = 0; i < 4; i++) {
      one_sweep.u_pass_histograms[i] = Alloc<unsigned int>(kRadix);
    }

    // Print how much memory we allocated in total

    size_t total = 0;
    total += n * sizeof(glm::vec4);
    total += n * sizeof(unsigned int);
    total += n * sizeof(unsigned int);
    total += kRadix * kRadixPasses * sizeof(unsigned int);
    total += kRadixPasses * sizeof(unsigned int);
    total += 4 * kRadix * sizeof(unsigned int);
    spdlog::info("Allocated {} MB", total / 1024 / 1024);
  }

  ~Pipe() {
    checkCudaErrors(cudaFree(u_points));
    checkCudaErrors(cudaFree(one_sweep.u_sort));
    checkCudaErrors(cudaFree(one_sweep.u_sort_alt));
    checkCudaErrors(cudaFree(one_sweep.u_global_histogram));
    checkCudaErrors(cudaFree(one_sweep.u_index));
    for (int i = 0; i < 4; i++) {
      checkCudaErrors(cudaFree(one_sweep.u_pass_histograms[i]));
    }
    spdlog::info("Freed memory");
  }

  const int n;
  int n_unique;
  int n_oct_nodes;

  // Component A: morton + sort
  glm::vec4* u_points;
  OneSweepData<4> one_sweep;

  // Component B: radix tree
  RadixTreeData brt;

  // Octree
  int* u_edge_count;
  int* u_prefix_sum;
  OctNode* u_oct_nodes;
};
