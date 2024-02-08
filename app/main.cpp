#include <omp.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <glm/glm.hpp>

#include "kernels/all.hpp"
#include "kernels/octree.hpp"

template <typename T>
void SaveToDataFile(const std::string &filename,
                    const T *data,
                    const int n,
                    bool binary = true) {
  auto filename_str = std::string(filename + (binary ? ".bin" : ".txt"));

  const auto bytes = n * sizeof(T);
  spdlog::info(
      "Saving {} mb to binary file: {}", bytes / (1024 * 1024), filename);

  std::ofstream out(filename_str, binary ? std::ios::binary : std::ios::out);
  if (!out.is_open()) {
    spdlog::error("Failed to open file");
    return;
  }
  if (binary) {
    out.write(reinterpret_cast<const char *>(data), n * sizeof(T));
  } else {
    for (auto i = 0; i < n; i++) {
      out << data[i] << '\n';
    }
  }
  out.close();
}

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
  k_InitRandomVec4Determinastic(u_input, n, min, range, seed);

  k_ComputeMortonCode(u_input, u_sort, n, min, range);

  k_SortKeysInplace(u_sort, n);

  const auto n_unique = k_CountUnique(u_sort, n);
  spdlog::info("n_unique = {}", n_unique);

  const auto tree = std::make_unique<RadixTree>(u_sort, n_unique, min, max);
  k_BuildRadixTree(tree.get());

  auto u_edge_count = new int[tree->n_nodes];

  k_EdgeCount(
      tree->d_tree.prefixN, tree->d_tree.parent, u_edge_count, n_unique);

  auto u_count_prefix_sum = new int[n_unique];

  [[maybe_unused]] auto _ =
      k_PartialSum(u_edge_count, 0, n_unique, u_count_prefix_sum);
  u_count_prefix_sum[0] = 0;

  const auto num_oct_nodes = u_count_prefix_sum[tree->n_nodes];
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
                 u_count_prefix_sum,
                 u_edge_count,
                 u_sort,
                 tree->d_tree.prefixN,
                 tree->d_tree.parent,
                 min,
                 range,
                 num_oct_nodes);

  k_LinkLeafNodes(u_oct_nodes,
                  u_count_prefix_sum,
                  u_edge_count,
                  u_sort,
                  tree->d_tree.hasLeafLeft,
                  tree->d_tree.hasLeafRight,
                  tree->d_tree.prefixN,
                  tree->d_tree.parent,
                  tree->d_tree.leftChild,
                  num_oct_nodes);

  constexpr bool save_as_binary = false;
  if (false) {
    SaveToDataFile("data/bm_sorted_mortons_u32_10m", u_sort, n, save_as_binary);
    SaveToDataFile("data/bm_prefix_n_u8_10m",
                   tree->d_tree.prefixN,
                   tree->n_nodes,
                   save_as_binary);
    SaveToDataFile("data/bm_parent_i32_10m",
                   tree->d_tree.parent,
                   tree->n_nodes,
                   save_as_binary);
    SaveToDataFile("data/bm_left_child_i32_10m",
                   tree->d_tree.leftChild,
                   tree->n_nodes,
                   save_as_binary);
    SaveToDataFile("data/bm_has_leaf_left_bool_10m",
                   tree->d_tree.hasLeafLeft,
                   tree->n_nodes,
                   save_as_binary);
    SaveToDataFile("data/bm_has_leaf_right_bool_10m",
                   tree->d_tree.hasLeafRight,
                   tree->n_nodes,
                   save_as_binary);
    SaveToDataFile("data/bm_edge_count_i32_10m",
                   u_edge_count,
                   tree->n_nodes,
                   save_as_binary);
    SaveToDataFile("data/bm_prefix_sum_i32_10m",
                   u_count_prefix_sum,
                   n_unique,
                   save_as_binary);
    // no need to save the oct, they will be computed at the very end
  }

  delete[] u_input;
  delete[] u_sort;
  delete[] u_edge_count;
  delete[] u_count_prefix_sum;
  delete[] u_oct_nodes;

  return 0;
}