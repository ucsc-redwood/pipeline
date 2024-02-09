#include <cuda_runtime.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <glm/glm.hpp>

#include "cuda/common.cuh"
#include "cuda/common/helper_cuda.hpp"
#include "kernels/all.hpp"

int main(const int argc, const char** argv) {
  int n = 10'000'000;
  int n_threads = 4;
  int my_num_blocks = 64;

  CLI::App app{"Multi-threaded sorting benchmark"};

  app.add_option("-n,--n", n, "Number of elements to sort")
      ->check(CLI::PositiveNumber);

  app.add_option("-t,--threads", n_threads, "Number of threads to use")
      ->check(CLI::Range(1, 48));

  app.add_option("-b,--blocks", my_num_blocks, "Number of blocks to use")
      ->check(CLI::PositiveNumber);

  CLI11_PARSE(app, argc, argv)

  spdlog::info("n = {}", n);
  spdlog::info("n_threads = {}", n_threads);

  omp_set_num_threads(n_threads);

  // ---------------------------------------------------------------------------

  unsigned int* u_sort;
  checkCudaErrors(cudaMallocManaged(&u_sort, n * sizeof(unsigned int)));

  gpu::k_InitAscendingSync(u_sort, n);

  // peek at the first 10 morton keys
  for (auto i = 0; i < 10; ++i) {
    spdlog::info("u_sort[{}] = {}", i, u_sort[i]);
  }

  const auto num_unique = n;
  const auto num_brt_nodes = num_unique - 1;

  // const auto block_size = DetermineBlockSize(gpu::k_BuildRadixTree_Kernel, n);
  // spdlog::info("block_size = {}", block_size);


  const auto block_size = 256;

  



  checkCudaErrors(cudaFree(u_sort));
  return 0;
}