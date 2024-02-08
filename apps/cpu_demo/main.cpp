#include <omp.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <glm/glm.hpp>

#include "kernels/all.hpp"
#include "types/brt.hpp"

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

  auto u_edge_count = new int[tree.n_nodes];

  k_EdgeCount(tree.prefixN, tree.parent, u_edge_count, n_unique);

  auto u_count_prefix_sum = new int[n_unique];

  [[maybe_unused]] auto _ =
      k_PartialSum(u_edge_count, 0, n_unique, u_count_prefix_sum);
  u_count_prefix_sum[0] = 0;

  const auto num_oct_nodes = u_count_prefix_sum[tree.n_nodes];
  spdlog::info("num_oct_nodes = {}", num_oct_nodes);

  auto u_oct_nodes = new OctNode[num_oct_nodes];

  const auto root_level = tree.prefixN[0] / 3;
  const auto root_prefix = u_sort[0] >> (morton_bits - (3 * root_level));

  cpu::morton32_to_xyz(&u_oct_nodes[0].cornor,
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

  delete[] u_input;
  delete[] u_sort;
  delete[] u_edge_count;
  delete[] u_count_prefix_sum;
  delete[] u_oct_nodes;

  return 0;
}