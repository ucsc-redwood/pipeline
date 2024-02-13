#pragma once

#include <cstdint>

struct RadixTreeData {
  int n_nodes;
  uint8_t *prefixN;
  bool *hasLeafLeft;
  bool *hasLeafRight;
  int *leftChild;
  int *parent;
};
