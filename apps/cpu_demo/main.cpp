#include <omp.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <glm/glm.hpp>

#include "kernels/all.hpp"
#include "types/brt.hpp"

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

  k_SortKeysInplace(u_sort, n);

  // peek 10 elements
  for (auto i = 0; i < 10; ++i) {
    spdlog::debug("u_sort[{}] = {}", i, u_sort[i]);
  }

  const auto n_unique = k_CountUnique(u_sort, n);
  spdlog::info("n_unique = {}", n_unique);

  return 0;
}