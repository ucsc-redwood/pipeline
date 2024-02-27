#pragma once

#include <glm/glm.hpp>

#include "helper.cuh"

struct OctNodes_better {
  explicit OctNodes_better(const size_t n) : n_nodes(n) {
    MallocManaged(&u_children, n * 8);
    MallocManaged(&u_corner, n);
    MallocManaged(&u_cell_size, n);
    MallocManaged(&u_child_node_mask, n);
    MallocManaged(&u_child_leaf_mask, n);
  }

  OctNodes_better(const OctNodes_better&) = delete;
  OctNodes_better& operator=(const OctNodes_better&) = delete;
  OctNodes_better(OctNodes_better&&) = delete;
  OctNodes_better& operator=(OctNodes_better&&) = delete;

  ~OctNodes_better() {
    CUDA_FREE(u_children);
    CUDA_FREE(u_corner);
    CUDA_FREE(u_cell_size);
    CUDA_FREE(u_child_node_mask);
    CUDA_FREE(u_child_leaf_mask);
  }

  void attachStream(const cudaStream_t stream) const {
    ATTACH_STREAM_SINGLE(u_children);
    ATTACH_STREAM_SINGLE(u_corner);
    ATTACH_STREAM_SINGLE(u_cell_size);
    ATTACH_STREAM_SINGLE(u_child_node_mask);
    ATTACH_STREAM_SINGLE(u_child_leaf_mask);
  }

  // for ~2m nodes, its about 71MB
  [[nodiscard]] size_t getMemorySize() const {
    size_t total = 0;
    total += n_nodes * 8 * sizeof(int);
    total += n_nodes * sizeof(glm::vec4);
    total += n_nodes * sizeof(float);
    total += n_nodes * sizeof(int);
    total += n_nodes * sizeof(int);
    return total;
  }

  const size_t n_nodes;

  int (*u_children)[8];
  glm::vec4* u_corner;
  float* u_cell_size;
  int* u_child_node_mask;
  int* u_child_leaf_mask;
};