#pragma once

#include <glm/glm.hpp>

// =============================================================================
// POD types
// The user should handle the memory allocation and deallocation (CPU/GPU)
// =============================================================================

/**
 * @brief All fields are "array"s of same length (number of brt nodes)
 *
 */
struct RadixTreeData {
  int n_nodes;
  uint8_t *prefixN;
  bool *hasLeafLeft;
  bool *hasLeafRight;
  int *leftChild;
  int *parent;
};

/**
 * @brief A single node in the octree. Currently, we have 8 children per node,
 * which is too much. But in the future, we might want to have a pointer.
 *
 */
struct OctNode {
  int children[8];
  glm::vec4 corner;
  float cell_size;
  int child_node_mask;
  int child_leaf_mask;
};

/**
 * @brief For OneSweep Radix Sort Algorithm (GPU only). All the intermediate
 * steps in the sorting algorithm.
 *
 */

template <int NUM_PASS>
struct OneSweepData {
  /**
   * @brief The number of elements to sort. E.g. 10M
   */
  unsigned int *u_sort;
  unsigned int *u_sort_alt;

  /**
   * @brief kRadix * kRadixPasses. E.g. 256 * 4 = 1024
   */
  unsigned int *u_global_histogram;

  /**
   * @brief kRadixPasses, e.g. 4
   */
  unsigned int *u_index;

  /**
   * @brief An array of pointers.
   * Each w/ kRadix * kRadixPasses, e.g. 256 * 4 (Each!)
   */
  unsigned int *u_pass_histograms[NUM_PASS];
};
