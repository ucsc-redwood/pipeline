#pragma once

#include <glm/glm.hpp>

namespace gpu {

struct OctNode {
  int children[8];
  glm::vec4 corner;
  float cell_size;
  int child_node_mask;
  int child_leaf_mask;

  __host__ __device__ __forceinline__ void SetChild(
      const unsigned int which_child, const int oct_idx) {
    children[which_child] = oct_idx;
    child_node_mask |= 1 << which_child;
  }

  __host__ __device__ __forceinline__ void SetLeaf(
      const unsigned int which_child, const int leaf_idx) {
    children[which_child] = leaf_idx;
    child_leaf_mask &= ~(1 << which_child);
  }
};

}  // namespace gpu
