#include <omp.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <glm/glm.hpp>

#include "kernels/02_sort.hpp"
#include "kernels/all.hpp"
#include "kernels/impl/morton.hpp"
#include "types/brt.hpp"

void checkTree(const unsigned int prefix,
               int code_len,
               const OctNode* nodes,
               const int oct_idx,
               const unsigned int* codes) {
  const OctNode& node = nodes[oct_idx];
  for (int i = 0; i < 8; ++i) {
    unsigned int new_pref = (prefix << 3) | i;
    if (node.child_node_mask & (1 << i)) {
      // print
      // printf("prefix = %u, code_len = %d, new_pref = %u, node.children[%d] =
      // %d\n", prefix, code_len, new_pref, i, node.children[i]);
      checkTree(new_pref, code_len + 3, nodes, node.children[i], codes);
    }
    if (node.child_leaf_mask & (1 << i)) {
      unsigned int leaf_prefix =
          codes[node.children[i]] >> (morton_bits - (code_len + 3));
      if (new_pref != leaf_prefix) {
        printf("oh no...\n");
      }
    }
  }
}

int main(const int argc, const char** argv) {
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

#ifdef DEBUG
  printf("DEBUG defined\n");
  spdlog::set_level(spdlog::level::debug);
#endif

  // ---------------------------------------------------------------------------

  constexpr auto min = 0.0f;
  constexpr auto max = 1024.0f;
  constexpr auto range = max - min;

  const auto u_input = new glm::vec4[n];
  const auto u_sort = new unsigned int[n];

  constexpr auto seed = 114514;

  k_InitRandomVec4(u_input, n, min, range, seed);

  k_ComputeMortonCode(u_input, u_sort, n, min, range);

  // k_SortKeysInplace(u_sort, n);
  k_SimpleRadixSort(u_sort, n);

  if (!std::is_sorted(u_sort, u_sort + n)) {
    spdlog::error("u_sort is not sorted");
    return 1;
  }

  // peek 10 elements
  for (auto i = 0; i < 10; ++i) {
    spdlog::debug("u_sort[{}] = {}", i, u_sort[i]);
  }

  const auto n_unique = k_CountUnique(u_sort, n);
  spdlog::info("n_unique = {}", n_unique);

  // Mannually allocate memory for radix tree (to make it consistent with the
  // CUDA implementation that uses 'cudaMallocManaged')
  RadixTreeData tree;
  tree.n_nodes = n_unique - 1;
  tree.prefixN = new uint8_t[tree.n_nodes];
  tree.hasLeafLeft = new bool[tree.n_nodes];
  tree.hasLeafRight = new bool[tree.n_nodes];
  tree.leftChild = new int[tree.n_nodes];
  tree.parent = new int[tree.n_nodes];

  k_BuildRadixTree(tree.n_nodes,
                   u_sort,
                   tree.prefixN,
                   tree.hasLeafLeft,
                   tree.hasLeafRight,
                   tree.leftChild,
                   tree.parent);

  const auto u_edge_count = new int[tree.n_nodes];

  k_EdgeCount(tree.prefixN, tree.parent, u_edge_count, n_unique);

  const auto u_count_prefix_sum = new int[n_unique];

  [[maybe_unused]] auto _ =
      k_PartialSum(u_edge_count, 0, n_unique, u_count_prefix_sum);
  u_count_prefix_sum[0] = 0;

  const auto num_oct_nodes = u_count_prefix_sum[tree.n_nodes];
  spdlog::info("num_oct_nodes = {}", num_oct_nodes);

  const auto u_oct_nodes = new OctNode[num_oct_nodes];

  const auto root_level = tree.prefixN[0] / 3;
  const auto root_prefix = u_sort[0] >> (morton_bits - (3 * root_level));

  cpu::morton32_to_xyz(&u_oct_nodes[0].corner,
                       root_prefix << (morton_bits - (3 * root_level)),
                       min,
                       range);
  u_oct_nodes[0].cell_size = range;

  k_MakeOctNodes(u_oct_nodes,
                 u_count_prefix_sum,
                 u_edge_count,
                 u_sort,
                 tree.prefixN,
                 tree.parent,
                 min,
                 range,
                 num_oct_nodes);

  k_LinkLeafNodes(u_oct_nodes,
                  u_count_prefix_sum,
                  u_edge_count,
                  u_sort,
                  tree.hasLeafLeft,
                  tree.hasLeafRight,
                  tree.prefixN,
                  tree.parent,
                  tree.leftChild,
                  num_oct_nodes);

  // ---------------------------------------------------------------------------

  // print 10 Nodes
  for (auto i = 0; i < 10; ++i) {
    spdlog::info(
        "u_oct_nodes[{}].corner = ({}, {}, {}), cell_size = {}, "
        "child_node_mask = {}, child_leaf_mask = {}",
        i,
        u_oct_nodes[i].corner.x,
        u_oct_nodes[i].corner.y,
        u_oct_nodes[i].corner.z,
        u_oct_nodes[i].cell_size,
        u_oct_nodes[i].child_node_mask,
        u_oct_nodes[i].child_leaf_mask);
  }

  delete[] u_input;
  delete[] u_sort;
  delete[] tree.prefixN;
  delete[] tree.hasLeafLeft;
  delete[] tree.hasLeafRight;
  delete[] tree.leftChild;
  delete[] tree.parent;
  delete[] u_edge_count;
  delete[] u_count_prefix_sum;
  delete[] u_oct_nodes;

  spdlog::info("CPU Demo Finished!");
  return 0;
}