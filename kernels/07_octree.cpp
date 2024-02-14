#include "kernels/07_octree.hpp"

#include "shared/oct.h"

void k_MakeOctNodes(OctNode* oct_nodes,
                    const int* node_offsets,    // prefix sum
                    const int* rt_node_counts,  // edge count
                    const unsigned int* codes,
                    const uint8_t* rt_prefixN,
                    const int* rt_parents,
                    const float min_coord,
                    const float range,
                    const int N  // number of brt nodes
) {
  // const auto root_level = rt_prefixN[0] / 3;

  // the root doesn't represent level 0 of the "entire" octree
#pragma omp parallel for
  for (auto i = 1; i < N; ++i) {
    shared::ProcessOctNode(i,
                           oct_nodes,
                           node_offsets,
                           rt_node_counts,
                           codes,
                           rt_prefixN,
                           rt_parents,
                           min_coord,
                           range,
                           N);
  }
}

void k_LinkLeafNodes(OctNode* nodes,
                     const int* node_offsets,
                     const int* rt_node_counts,
                     const unsigned int* codes,
                     const bool* rt_hasLeafLeft,
                     const bool* rt_hasLeafRight,
                     const uint8_t* rt_prefixN,
                     const int* rt_parents,
                     const int* rt_leftChild,
                     const int N) {
#pragma omp parallel for
  for (auto i = 0; i < N; i++) {
    shared::ProcessLinkLeaf(i,
                            nodes,
                            node_offsets,
                            rt_node_counts,
                            codes,
                            rt_hasLeafLeft,
                            rt_hasLeafRight,
                            rt_prefixN,
                            rt_parents,
                            rt_leftChild,
                            N);
  }
}
