#pragma once

#include <benchmark/benchmark.h>
#include <omp.h>

#include <glm/glm.hpp>

#include "types/brt.hpp"

namespace bm = benchmark;

#include "config.hpp"
#include "kernels/all.hpp"

template <typename T>
[[nodiscard]] T* AllocateHost(const size_t n_items) {
  return static_cast<T*>(malloc(n_items * sizeof(T)));
}

inline void Free(void* ptr) { free(ptr); }

// Fixture for all benchmarks
// Basically, run the full application one time. The results of each stage will
// be stored in the fixture. You should treat them as read-only.
// Put the output to somewhere else
class MyFixture : public bm::Fixture {
 public:
  MyFixture() {
    u_input = AllocateHost<glm::vec4>(kN);
    u_morton = AllocateHost<unsigned int>(kN);

    k_InitRandomVec4(u_input, kN, kMin, kRange, kRandomSeed);
    k_ComputeMortonCode(u_input, u_morton, kN, kMin, kRange);

    // Better sort and unique thank our kernels.
    std::sort(u_morton, u_morton + kN);
    const auto it = std::unique(u_morton, u_morton + kN);
    n_unique = std::distance(u_morton, it);

    tree.n_nodes = n_unique - 1;
    tree.prefixN = AllocateHost<uint8_t>(tree.n_nodes);
    tree.hasLeafLeft = AllocateHost<bool>(tree.n_nodes);
    tree.hasLeafRight = AllocateHost<bool>(tree.n_nodes);
    tree.leftChild = AllocateHost<int>(tree.n_nodes);
    tree.parent = AllocateHost<int>(tree.n_nodes);

    k_BuildRadixTree(tree.n_nodes,
                     u_morton,
                     tree.prefixN,
                     tree.hasLeafLeft,
                     tree.hasLeafRight,
                     tree.leftChild,
                     tree.parent);

    u_edge_count = AllocateHost<int>(tree.n_nodes);
    u_count_prefix_sum = AllocateHost<int>(n_unique);

    k_EdgeCount(tree.prefixN, tree.parent, u_edge_count, n_unique);

    [[maybe_unused]] auto _ =
        k_PartialSum(u_edge_count, 0, n_unique, u_count_prefix_sum);
    u_count_prefix_sum[0] = 0;

    n_oct_nodes = u_count_prefix_sum[tree.n_nodes];

    // (Verified already)
    // printf("==============================n_oct_nodes = %d\n", n_oct_nodes);

    // ----------------- used by output -----------------
    u_morton_out = AllocateHost<unsigned int>(kN);
    tree_out.n_nodes = tree.n_nodes;
    tree_out.prefixN = AllocateHost<uint8_t>(tree.n_nodes);
    tree_out.hasLeafLeft = AllocateHost<bool>(tree.n_nodes);
    tree_out.hasLeafRight = AllocateHost<bool>(tree.n_nodes);
    tree_out.leftChild = AllocateHost<int>(tree.n_nodes);
    tree_out.parent = AllocateHost<int>(tree.n_nodes);
    u_edge_count_out = AllocateHost<int>(tree.n_nodes);
    u_count_prefix_sum_out = AllocateHost<int>(n_unique);
  }

  ~MyFixture() {
    Free(u_input);
    Free(u_morton);
    Free(tree.prefixN);
    Free(tree.hasLeafLeft);
    Free(tree.hasLeafRight);
    Free(tree.leftChild);
    Free(tree.parent);
    Free(u_edge_count);
    Free(u_count_prefix_sum);

    Free(u_morton_out);
    Free(tree_out.prefixN);
    Free(tree_out.hasLeafLeft);
    Free(tree_out.hasLeafRight);
    Free(tree_out.leftChild);
    Free(tree_out.parent);
    Free(u_edge_count_out);
    Free(u_count_prefix_sum_out);
  }

  int n_unique;
  int n_oct_nodes;

  // These should be treated as read-only
  glm::vec4* u_input;
  unsigned int* u_morton;
  RadixTreeData tree;
  int* u_edge_count;
  int* u_count_prefix_sum;

  // You may use these as output
  unsigned int* u_morton_out;
  RadixTreeData tree_out;
  int* u_edge_count_out;
  int* u_count_prefix_sum_out;
};
