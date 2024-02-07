#include <omp.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <glm/glm.hpp>

#include "kernels/init.hpp"
#include "kernels/morton.hpp"
#include "kernels/sort.hpp"

int main(const int argc, const char **argv) {
  int n = 10'000'000;
  int n_threads = 4;

  CLI::App app{"Multi-threaded sorting benchmark"};

  app.add_option("-n,--n", n, "Number of elements to sort")
      ->check(CLI::PositiveNumber);

  app.add_option("-t,--threads", n_threads, "Number of threads to use")
      ->check(CLI::Range(1, 48));

  CLI11_PARSE(app, argc, argv);

  spdlog::info("n = {}", n);
  spdlog::info("n_threads = {}", n_threads);

  omp_set_num_threads(n_threads);

  // ---------------------------------------------------------------------------

  constexpr auto min = 0.0f;
  constexpr auto max = 1024.0f;
  constexpr auto range = max - min;

  auto u_input = new glm::vec4[n];
  auto u_sort = new unsigned int[n];

  constexpr auto seed = 114514;
  k_InitRandomVec4(u_input, n, min, range, seed);

  // peek the first 10 elements
  for (auto i = 0; i < 10; i++) {
    spdlog::info("u_input[{}] = ({}, {}, {}, {})",
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
    spdlog::info("u_sort[{}] = {}", i, u_sort[i]);
  }
  
  delete[] u_input;
  delete[] u_sort;

  return 0;
}