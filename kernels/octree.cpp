#include "kernels/octree.hpp"

#include <cstdio>

#include "kernels/morton.hpp"

void k_MakeOctNodes(OctNode* oct_nodes,
                    const int* node_offsets,    // prefix sum
                    const int* rt_node_counts,  // edge count
                    const MortonT* codes,
                    const uint8_t* rt_prefixN,
                    const int* rt_parents,
                    float min_coord,
                    float range,
                    int N  // number of brt nodes
) {
  const auto root_level = rt_prefixN[0] / 3;

  // the root doesn't represent level 0 of the "entire" octree
#pragma omp parallel for
  for (auto i = 1; i < N; ++i) {
    auto oct_idx = node_offsets[i];
    const auto n_new_nodes = rt_node_counts[i];

    // for each nodes, make n_new_nodes
    for (auto j = 0; j < n_new_nodes - 1; ++j) {
      const auto level = rt_prefixN[i] / 3 - j;

      const auto node_prefix = codes[i] >> (morton_bits - (3 * level));
      const auto which_child = node_prefix & 0b111;
      const auto parent = oct_idx + 1;

      // oct_nodes[parent].children[which_child] = oct_idx;

      oct_nodes[parent].SetChild(which_child, oct_idx);

      cpu::morton32_to_xyz(&oct_nodes[oct_idx].cornor,
                           node_prefix << (morton_bits - (3 * level)),
                           min_coord,
                           range);

      // each cell is half the size of the level above it
      oct_nodes[oct_idx].cell_size = range / (float)(1 << (level - root_level));
      oct_idx = parent;
    }

    if (n_new_nodes > 0) {
      auto rt_parent = rt_parents[i];

      auto counter = 0;
      while (rt_node_counts[rt_parent] == 0) {
        rt_parent = rt_parents[rt_parent];

        ++counter;
        if (counter > 30) {  // 64 / 3
          // tb_trace_w("counter > 22");
          break;
        }
      }

      const auto oct_parent = node_offsets[rt_parent];
      const auto top_level = rt_prefixN[i] / 3 - n_new_nodes + 1;
      const auto top_node_prefix = codes[i] >> (morton_bits - (3 * top_level));

      const auto which_child = top_node_prefix & 0b111;

      // oct_nodes[oct_parent].children[which_child] = oct_idx;
      // oct_nodes[oct_parent].SetChild(oct_idx, which_child);
      oct_nodes[oct_parent].SetChild(which_child, oct_idx);

      cpu::morton32_to_xyz(&oct_nodes[oct_idx].cornor,
                           top_node_prefix << (morton_bits - (3 * top_level)),
                           min_coord,
                           range);
      oct_nodes[oct_idx].cell_size =
          range / (float)(1 << (top_level - root_level));
    }
  }
}

void k_LinkLeafNodes(OctNode* nodes,
                     const int* node_offsets,
                     const int* rt_node_counts,
                     const MortonT* codes,
                     const bool* rt_hasLeafLeft,
                     const bool* rt_hasLeafRight,
                     const uint8_t* rt_prefixN,
                     const int* rt_parents,
                     const int* rt_leftChild,
                     const int N) {
#pragma omp parallel for
  for (auto i = 0; i < N; i++) {
    if (rt_hasLeafLeft[i]) {
      int leaf_idx = rt_leftChild[i];
      int leaf_level = rt_prefixN[i] / 3 + 1;
      MortonT leaf_prefix = codes[leaf_idx] >> (morton_bits - (3 * leaf_level));
      int which_child = leaf_prefix & 0b111;
      // walk up the radix tree until finding a node which contributes an
      // octnode
      int rt_node = i;
      while (rt_node_counts[rt_node] == 0) {
        rt_node = rt_parents[rt_node];
      }
      // the lowest octnode in the string contributed by rt_node will be the
      // lowest index
      int bottom_oct_idx = node_offsets[rt_node];
      nodes[bottom_oct_idx].SetLeaf(which_child, leaf_idx);
    }
    if (rt_hasLeafRight[i]) {
      int leaf_idx = rt_leftChild[i] + 1;
      int leaf_level = rt_prefixN[i] / 3 + 1;
      MortonT leaf_prefix = codes[leaf_idx] >> (morton_bits - (3 * leaf_level));
      int which_child = leaf_prefix & 0b111;
      int rt_node = i;
      while (rt_node_counts[rt_node] == 0) {
        rt_node = rt_parents[rt_node];
      }
      // the lowest octnode in the string contributed by rt_node will be the
      // lowest index
      int bottom_oct_idx = node_offsets[rt_node];
      nodes[bottom_oct_idx].SetLeaf(which_child, leaf_idx);
    }
  }
}
