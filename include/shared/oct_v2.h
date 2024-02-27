#pragma once

#include "defines.h"
#include "morton.h"
// #include "types.h"

namespace shared {

namespace v2 {

H_D_I void SetChild(const int node_idx,
                    int (*u_children)[8],
                    int* u_child_node_mask,
                    const unsigned int which_child,
                    const int oct_idx) {
  u_children[node_idx][which_child] = oct_idx;
  u_child_node_mask[node_idx] |= 1 << which_child;
}

H_D_I void SetLeaf(const int node_idx,
                   int (*u_children)[8],
                   int* u_child_leaf_mask,
                   const unsigned int which_child,
                   const int leaf_idx) {
  u_children[node_idx][which_child] = leaf_idx;
  u_child_leaf_mask[node_idx] &= ~(1 << which_child);
}

// int (*u_children)[8];
// glm::vec4* u_corner;
// float* u_cell_size;
// int* u_child_node_mask;
// int* u_child_leaf_mask;

// processing for index 'i'
H_D_I void ProcessOctNode(const int i /*brt node index*/,
                          // --------------------------
                          int (*u_children)[8],
                          glm::vec4* u_corner,
                          float* u_cell_size,
                          int* u_child_node_mask,
                          // --------------------------
                          const int* node_offsets,    // prefix sum
                          const int* rt_node_counts,  // edge count
                          const unsigned int* codes,
                          const uint8_t* rt_prefixN,
                          const int* rt_parents,
                          const float min_coord,
                          const float range) {
  // For octrees, it starts at 'offset[x]', and the numbers is decided by the
  // 'count[i]'. You can imagine something like:
  // brt[0] contains oct nodes [0, 3] (4 total)
  // brt[1] contains oct nodes [4, 4] (1 total)
  // brt[2] contains oct nodes [5, 6] (2 total) ...
  auto oct_idx = node_offsets[i];
  const auto n_new_nodes = rt_node_counts[i];

  // just a constant
  const auto root_level = rt_prefixN[0] / 3;

  // for each new node,
  // (1) create their cornor/cell size
  // (2) attach them to their parent
  for (auto j = 0; j < n_new_nodes - 1; ++j) {
    const auto level = rt_prefixN[i] / 3 - j;  // every new node has a level

    const auto node_prefix = codes[i] >> (morton_bits - (3 * level));
    const auto which_child = node_prefix & 0b111;
    const auto parent = oct_idx + 1;

    // set the parent's child to the current octnode
    SetChild(parent, u_children, u_child_node_mask, which_child, oct_idx);

    // compute the corner of the current octnode
    morton32_to_xyz(&u_corner[oct_idx],
                    node_prefix << (morton_bits - (3 * level)),
                    min_coord,
                    range);

    // each cell is half the size of the level above it
    u_cell_size[oct_idx] =
        range / static_cast<float>(1 << (level - root_level));

    // go to the next octnode (parent)
    oct_idx = parent;
  }

  if (n_new_nodes > 0) {
    auto rt_parent = rt_parents[i];

    auto counter = 0;
    while (rt_node_counts[rt_parent] == 0) {
      rt_parent = rt_parents[rt_parent];

      ++counter;
      if (counter > 30) {  // 64 / 3
        break;
      }
    }

    const auto oct_parent = node_offsets[rt_parent];
    const auto top_level = rt_prefixN[i] / 3 - n_new_nodes + 1;
    const auto top_node_prefix = codes[i] >> (morton_bits - (3 * top_level));

    const auto which_child = top_node_prefix & 0b111;

    SetChild(oct_parent, u_children, u_child_node_mask, which_child, oct_idx);

    morton32_to_xyz(&u_corner[oct_idx],
                    top_node_prefix << (morton_bits - (3 * top_level)),
                    min_coord,
                    range);

    u_cell_size[oct_idx] =
        range / static_cast<float>(1 << (top_level - root_level));
  }
}

H_D_I void ProcessLinkLeaf(const int i /*brt node index*/,
                           // --------------------------
                           int (*u_children)[8],
                           int* u_child_leaf_mask,
                           // --------------------------
                           const int* node_offsets,
                           const int* rt_node_counts,
                           const unsigned int* codes,
                           const bool* rt_hasLeafLeft,
                           const bool* rt_hasLeafRight,
                           const uint8_t* rt_prefixN,
                           const int* rt_parents,
                           const int* rt_leftChild) {
  if (rt_hasLeafLeft[i]) {
    auto leaf_idx = rt_leftChild[i];
    auto leaf_level = rt_prefixN[i] / 3 + 1;
    auto leaf_prefix = codes[leaf_idx] >> (morton_bits - (3 * leaf_level));
    auto child_idx = leaf_prefix & 0b111;
    // walk up the radix tree until finding a node which contributes an octnode
    auto rt_node = i;
    while (rt_node_counts[rt_node] == 0) {
      rt_node = rt_parents[rt_node];
    }
    // the lowest octnode in the string contributed by rt_node will be the
    // lowest index
    auto bottom_oct_idx = node_offsets[rt_node];

    // SetLeaf(&nodes[bottom_oct_idx], child_idx, leaf_idx);
    SetLeaf(bottom_oct_idx, u_children, u_child_leaf_mask, child_idx, leaf_idx);
  }
  if (rt_hasLeafRight[i]) {
    auto leaf_idx = rt_leftChild[i] + 1;
    auto leaf_level = rt_prefixN[i] / 3 + 1;
    auto leaf_prefix = codes[leaf_idx] >> (morton_bits - (3 * leaf_level));
    auto child_idx = leaf_prefix & 0b111;
    auto rt_node = i;
    while (rt_node_counts[rt_node] == 0) {
      rt_node = rt_parents[rt_node];
    }
    // the lowest octnode in the string contributed by rt_node will be the
    // lowest index
    auto bottom_oct_idx = node_offsets[rt_node];
    // SetLeaf(&nodes[bottom_oct_idx], child_idx, leaf_idx);
    SetLeaf(bottom_oct_idx, u_children, u_child_leaf_mask, child_idx, leaf_idx);
  }
}

}  // namespace v2

}  // namespace shared
