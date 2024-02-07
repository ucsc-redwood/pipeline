#include <omp.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <glm/glm.hpp>

#include "kernels/all.hpp"
#include "kernels/octree.hpp"

int main(const int argc, const char **argv) {
  int n = 10'000'000;
  int n_threads = 4;

  CLI::App app{"Multi-threaded sorting benchmark"};

  app.add_option("-n,--n", n, "Number of elements to sort")
      ->check(CLI::PositiveNumber);

  app.add_option("-t,--threads", n_threads, "Number of threads to use")
      ->check(CLI::Range(1, 48));

  CLI11_PARSE(app, argc, argv)

  spdlog::info("n = {}", n);
  spdlog::info("n_threads = {}", n_threads);

  omp_set_num_threads(n_threads);

  // ---------------------------------------------------------------------------

  constexpr auto min = 0.0f;
  constexpr auto max = 1024.0f;
  constexpr auto range = max - min;

  const auto u_input = new glm::vec4[n];
  auto u_sort = new unsigned int[n];

  constexpr auto seed = 114514;
  k_InitRandomVec4(u_input, n, min, range, seed);

  // peek the first 10 elements
  for (auto i = 0; i < 10; i++) {
    spdlog::debug("u_input[{}] = ({}, {}, {}, {})",
                  i,
                  u_input[i].x,
                  u_input[i].y,
                  u_input[i].z,
                  u_input[i].w);
  }

  k_ComputeMortonCode(u_input, u_sort, n, min, range);

  k_SortKeysInplace(u_sort, n);

  // peek the first 10 elements
  for (auto i = 0; i < 10; i++) {
    spdlog::debug("u_sort[{}] = {}", i, u_sort[i]);
  }

  const auto n_unique = k_CountUnique(u_sort, n);
  spdlog::info("n_unique = {}", n_unique);
  spdlog::info("n - n_unique = {}", n - n_unique);

  const auto tree = std::make_unique<RadixTree>(u_sort, n_unique, min, max);
  k_BuildRadixTree(tree.get());

  for (auto i = 0; i < 32; ++i) {
    printf(
        "idx = %d, code = %u, prefixN = %d, left = %d, parent = %d, "
        "leftLeaf=%d, rightLeft=%d\n",
        i,
        u_sort[i],
        tree->d_tree.prefixN[i],
        tree->d_tree.leftChild[i],
        tree->d_tree.parent[i],
        tree->d_tree.hasLeafLeft[i],
        tree->d_tree.hasLeafRight[i]);
  }

  auto u_edge_count = new int[tree->n_nodes];

  k_EdgeCount(
      tree->d_tree.prefixN, tree->d_tree.parent, u_edge_count, n_unique);

  // peek the first 10 elements
  for (auto i = 0; i < 10; i++) {
    spdlog::debug("u_edge_count[{}] = {}", i, u_edge_count[i]);
  }

  auto u_count_prefox_sum = new int[n_unique];

  [[maybe_unused]] auto _ =
      k_PartialSum(u_edge_count, 0, n_unique, u_count_prefox_sum);
  u_count_prefox_sum[0] = 0;

  // peek the first 10 elements
  for (auto i = 0; i < 10; i++) {
    spdlog::info("u_count_prefox_sum[{}] = {}", i, u_count_prefox_sum[i]);
  }

  const auto num_oct_nodes = u_count_prefox_sum[tree->n_nodes];
  spdlog::info("num_oct_nodes = {}", num_oct_nodes);

  auto u_oct_nodes = new OctNode[num_oct_nodes];
  const auto root_level = tree->d_tree.prefixN[0] / 3;

  const auto root_prefix = u_sort[0] >> (morton_bits - (3 * root_level));

  morton32_to_xyz(&u_oct_nodes[0].cornor,
                  root_prefix << (morton_bits - (3 * root_level)),
                  min,
                  range);
  u_oct_nodes[0].cell_size = range;

  k_MakeOctNodes(u_oct_nodes,
                 u_count_prefox_sum,
                 u_edge_count,
                 u_sort,
                 tree->d_tree.prefixN,
                 tree->d_tree.parent,
                 min,
                 range,
                 num_oct_nodes);
  // peek 32 octree nodes
  for (auto i = 0; i < 32; ++i) {
    printf("idx = %d, parent = %d, cell_size = %f, corner = (%f, %f, %f)\n",
           i,
           u_oct_nodes[i].child_node_mask,
           u_oct_nodes[i].cell_size,
           u_oct_nodes[i].cornor.x,
           u_oct_nodes[i].cornor.y,
           u_oct_nodes[i].cornor.z);
  }

  k_LinkLeafNodes(u_oct_nodes,
                  u_count_prefox_sum,
                  u_edge_count,
                  u_sort,
                  tree->d_tree.hasLeafLeft,
                  tree->d_tree.hasLeafRight,
                  tree->d_tree.prefixN,
                  tree->d_tree.parent,
                  tree->d_tree.leftChild,
                  num_oct_nodes);

  checkTree(root_prefix, root_level, u_oct_nodes, 0, u_sort);

  delete[] u_input;
  delete[] u_sort;
  delete[] u_edge_count;
  delete[] u_count_prefox_sum;
  delete[] u_oct_nodes;

  return 0;
}