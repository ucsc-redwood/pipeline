#include <omp.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <glm/glm.hpp>

#include "kernels/init.hpp"
#include "kernels/morton.hpp"
#include "kernels/radix_tree.hpp"
#include "kernels/sort.hpp"
#include "kernels/unique.hpp"

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

  delete[] u_input;
  delete[] u_sort;

  return 0;
}