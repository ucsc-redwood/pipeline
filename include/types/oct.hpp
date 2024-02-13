#pragma once

#include <glm/glm.hpp>

struct OctNode {
  int children[8];
  glm::vec4 corner;
  float cell_size;
  int child_node_mask;
  int child_leaf_mask;
};
