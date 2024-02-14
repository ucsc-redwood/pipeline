#pragma once

#include <glm/glm.hpp>

struct RadixTreeData {
  int n_nodes;
  uint8_t *prefixN;
  bool *hasLeafLeft;
  bool *hasLeafRight;
  int *leftChild;
  int *parent;
};

struct OctNode {
  int children[8];
  glm::vec4 corner;
  float cell_size;
  int child_node_mask;
  int child_leaf_mask;
};
