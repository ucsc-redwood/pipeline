#include "kernels/edge_count.hpp"

void k_EdgeCount(const uint8_t* prefix_n,
                 int* parents,
                 int* edge_count,
                 const int n_brt_nodes) {
  edge_count[0] = 0;

  // root has no parent, so don't do for index 0
#pragma omp parallel for schedule(static)
  for (auto i = 1; i < n_brt_nodes; ++i) {
    const auto my_depth = prefix_n[i] / 3;
    const auto parent_depth = prefix_n[parents[i]] / 3;
    edge_count[i] = my_depth - parent_depth;
  }
}