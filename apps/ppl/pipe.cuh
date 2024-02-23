#pragma once

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <glm/glm.hpp>

#include "cuda/common/helper_cuda.hpp"
#include "gpu_kernels.cuh"
#include "shared/types.h"

struct Pipe {
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
    one_sweep = gpu::OneSweepHelper::CreateOnesweepData(one_sweep, n);

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
    gpu::OneSweepHelper::DestroyOnesweepData(one_sweep);


    spdlog::info("Freed memory");
  }

  const int n;
  int n_unique;
  int n_oct_nodes;

  // Component A: morton + sort
  glm::vec4* u_points;
  OneSweepData<4> one_sweep;
  RadixTreeData brt;
  int* u_edge_count;
  int* u_prefix_sum;
  OctNode* u_oct_nodes;
};
