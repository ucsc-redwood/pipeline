#pragma once

#include "defines.h"
#include "morton.h"
#include "types.h"

namespace shared {

H_D_I void SetChild(OctNode* node,
                    const unsigned int which_child,
                    const int oct_idx) {
  node->children[which_child] = oct_idx;
  node->child_node_mask |= 1 << which_child;
}

H_D_I void SetLeaf(OctNode* node,
                   const unsigned int which_child,
                   const int leaf_idx) {
  node->children[which_child] = leaf_idx;
  node->child_leaf_mask &= ~(1 << which_child);
}

// processing for index 'i'
H_D_I void ProcessOctNode(const int i,
                          OctNode* oct_nodes,
                          const int* node_offsets,    // prefix sum
                          const int* rt_node_counts,  // edge count
                          const unsigned int* codes,
                          const uint8_t* rt_prefixN,
                          const int* rt_parents,
                          const float min_coord,
                          const float range,
                          const int N  // number of brt nodes
) {
  auto oct_idx = node_offsets[i];
  const auto n_new_nodes = rt_node_counts[i];

  const auto root_level = rt_prefixN[0] / 3;

  // for each nodes, make n_new_nodes
  for (auto j = 0; j < n_new_nodes - 1; ++j) {
    const auto level = rt_prefixN[i] / 3 - j;

    const auto node_prefix = codes[i] >> (morton_bits - (3 * level));
    const auto which_child = node_prefix & 0b111;
    const auto parent = oct_idx + 1;

    SetChild(&oct_nodes[parent], which_child, oct_idx);

    morton32_to_xyz(&oct_nodes[oct_idx].corner,
                    node_prefix << (morton_bits - (3 * level)),
                    min_coord,
                    range);

    // each cell is half the size of the level above it
    oct_nodes[oct_idx].cell_size =
        range / static_cast<float>(1 << (level - root_level));
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

    SetChild(&oct_nodes[oct_parent], which_child, oct_idx);

    morton32_to_xyz(&oct_nodes[oct_idx].corner,
                    top_node_prefix << (morton_bits - (3 * top_level)),
                    min_coord,
                    range);

    oct_nodes[oct_idx].cell_size =
        range / static_cast<float>(1 << (top_level - root_level));
  }
}

H_D_I void ProcessLinkLeaf(int i,
                           OctNode* nodes,
                           const int* node_offsets,
                           const int* rt_node_counts,
                           const unsigned int* codes,
                           const bool* rt_hasLeafLeft,
                           const bool* rt_hasLeafRight,
                           const uint8_t* rt_prefixN,
                           const int* rt_parents,
                           const int* rt_leftChild,
                           const int N) {
  if (rt_hasLeafLeft[i]) {
    int leaf_idx = rt_leftChild[i];
    int leaf_level = rt_prefixN[i] / 3 + 1;
    unsigned int leaf_prefix =
        codes[leaf_idx] >> (morton_bits - (3 * leaf_level));
    int child_idx = leaf_prefix & 0b111;
    // walk up the radix tree until finding a node which contributes an octnode
    int rt_node = i;
    while (rt_node_counts[rt_node] == 0) {
      rt_node = rt_parents[rt_node];
    }
    // the lowest octnode in the string contributed by rt_node will be the
    // lowest index
    int bottom_oct_idx = node_offsets[rt_node];
    SetLeaf(&nodes[bottom_oct_idx], child_idx, leaf_idx);
  }
  if (rt_hasLeafRight[i]) {
    int leaf_idx = rt_leftChild[i] + 1;
    int leaf_level = rt_prefixN[i] / 3 + 1;
    unsigned int leaf_prefix =
        codes[leaf_idx] >> (morton_bits - (3 * leaf_level));
    int child_idx = leaf_prefix & 0b111;
    int rt_node = i;
    while (rt_node_counts[rt_node] == 0) {
      rt_node = rt_parents[rt_node];
    }
    // the lowest octnode in the string contributed by rt_node will be the
    // lowest index
    int bottom_oct_idx = node_offsets[rt_node];
    SetLeaf(&nodes[bottom_oct_idx], child_idx, leaf_idx);
  }
}

}  // namespace shared
