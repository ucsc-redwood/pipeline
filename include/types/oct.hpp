#pragma once

#include <glm/glm.hpp>

struct OctNode {
  int children[8];
  glm::vec4 cornor;
  float cell_size;
  int child_node_mask;
  int child_leaf_mask;

  void SetChild(int which_child, int oct_idx) {
    children[which_child] = oct_idx;
    child_node_mask |= 1 << which_child;
  }

  void SetLeaf(int which_child, int leaf_idx) {
    children[which_child] = leaf_idx;
    child_leaf_mask &= ~(1 << which_child);
  }
};
