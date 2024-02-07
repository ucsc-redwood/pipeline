#pragma once

#include <glm/glm.hpp>

struct OctNode {
  int children[8];
  glm::vec4 cornor;
  float cell_size;
  int child_node_mask;
  int child_leaf_mask;

  void SetChild(int index, int child) {
    children[index] = child;
    child_node_mask |= 1 << index;
  }

  void SetLeaf(int index, int leaf) {
    children[index] = leaf;
    child_leaf_mask &= ~(1 << index);
  }
};
