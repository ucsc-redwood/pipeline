#pragma once

#include "helper.cuh"

struct RadixTree {
  explicit RadixTree(const int n) : n_nodes(n) {
    MallocManaged(&prefixN, n);
    MallocManaged(&hasLeafLeft, n);
    MallocManaged(&hasLeafRight, n);
    MallocManaged(&leftChild, n);
    MallocManaged(&parent, n);
  }

  RadixTree(const RadixTree&) = delete;
  RadixTree& operator=(const RadixTree&) = delete;
  RadixTree(RadixTree&&) = delete;
  RadixTree& operator=(RadixTree&&) = delete;

  ~RadixTree() {
    CUDA_FREE(prefixN);
    CUDA_FREE(hasLeafLeft);
    CUDA_FREE(hasLeafRight);
    CUDA_FREE(leftChild);
    CUDA_FREE(parent);
  }

  void attachStream(const cudaStream_t stream) const {
    ATTACH_STREAM_SINGLE(prefixN);
    ATTACH_STREAM_SINGLE(hasLeafLeft);
    ATTACH_STREAM_SINGLE(hasLeafRight);
    ATTACH_STREAM_SINGLE(leftChild);
    ATTACH_STREAM_SINGLE(parent);
  }

  // For ~2m nodes, its about ~21MB
  [[nodiscard]] size_t getMemorySize() const {
    size_t total = 0;
    total += n_nodes * sizeof(uint8_t);
    total += n_nodes * sizeof(bool);
    total += n_nodes * sizeof(bool);
    total += n_nodes * sizeof(int);
    total += n_nodes * sizeof(int);
    return total;
  }

  const int n_nodes;
  uint8_t* prefixN;
  bool* hasLeafLeft;
  bool* hasLeafRight;
  int* leftChild;
  int* parent;
};
