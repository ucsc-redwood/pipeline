#pragma once

#include <glm/glm.hpp>

struct OctNodes_better {
  explicit OctNodes_better(const int n) : n_nodes(n) {
    // clang-format off
    cudaMallocManaged(reinterpret_cast<void**>(&u_children), n * 8 * sizeof(int));
    cudaMallocManaged(reinterpret_cast<void**>(&u_corner), n * sizeof(glm::vec4));
    cudaMallocManaged(reinterpret_cast<void**>(&u_cell_size), n * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&u_child_node_mask), n * sizeof(int));
    cudaMallocManaged(reinterpret_cast<void**>(&u_child_leaf_mask), n * sizeof(int));
    // clang-format on
  }

  ~OctNodes_better() {
    cudaFree(u_children);
    cudaFree(u_corner);
    cudaFree(u_cell_size);
    cudaFree(u_child_node_mask);
    cudaFree(u_child_leaf_mask);
  }

  void attachStream(const cudaStream_t stream) const {
    cudaStreamAttachMemAsync(stream, u_children, 0, cudaMemAttachSingle);
    cudaStreamAttachMemAsync(stream, u_corner, 0, cudaMemAttachSingle);
    cudaStreamAttachMemAsync(stream, u_cell_size, 0, cudaMemAttachSingle);
    cudaStreamAttachMemAsync(stream, u_child_node_mask, 0, cudaMemAttachSingle);
    cudaStreamAttachMemAsync(stream, u_child_leaf_mask, 0, cudaMemAttachSingle);
  }

  [[nodiscard]] size_t getMemorySize() const {
    size_t total = 0;
    total += n_nodes * 8 * sizeof(int);
    total += n_nodes * sizeof(glm::vec4);
    total += n_nodes * sizeof(float);
    total += n_nodes * sizeof(int);
    total += n_nodes * sizeof(int);
    return total;
  }

  int n_nodes;
  // int* u_num_nodes;

  int (*u_children)[8];
  glm::vec4* u_corner;
  float* u_cell_size;
  int* u_child_node_mask;
  int* u_child_leaf_mask;
};