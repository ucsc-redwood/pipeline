#pragma once

#include <cstdint>

struct RadixTreeData {
  // tmp
  ~RadixTreeData() {
    delete[] prefixN;
    delete[] hasLeafLeft;
    delete[] hasLeafRight;
    delete[] leftChild;
    delete[] parent;
  }

  int n_nodes;
  uint8_t *prefixN;
  bool *hasLeafLeft;
  bool *hasLeafRight;
  int *leftChild;
  int *parent;
};
