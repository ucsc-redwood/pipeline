#include <cuda_runtime.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <glm/glm.hpp>

#include "cuda/common.cuh"
#include "cuda/common/helper_cuda.hpp"
#include "cuda/kernels/init.cuh"
#include "cuda/kernels/radix_tree.cuh"
#include "kernels/all.hpp"
#include "types/brt.hpp"

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

  RadixTreeData radix_data;
  radix_data.n_nodes = num_unique - 1;
  // clang-format off
  checkCudaErrors(cudaMallocManaged(&radix_data.prefixN, num_unique * sizeof(uint8_t)));
  checkCudaErrors(cudaMallocManaged(&radix_data.hasLeafLeft, num_unique * sizeof(bool)));
  checkCudaErrors(cudaMallocManaged(&radix_data.hasLeafRight, num_unique * sizeof(bool)));
  checkCudaErrors(cudaMallocManaged(&radix_data.leftChild, num_unique * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(&radix_data.parent, num_unique * sizeof(int)));
  // clang-format on

  const auto block_size = DetermineBlockSize(gpu::k_BuildRadixTree_Kernel);
  spdlog::info("block_size = {}", block_size);

  gpu::k_BuildRadixTree_Kernel<<<my_num_blocks, block_size>>>(
      num_unique,
      u_sort,
      radix_data.prefixN,
      radix_data.hasLeafLeft,
      radix_data.hasLeafRight,
      radix_data.leftChild,
      radix_data.parent);
  checkCudaErrors(cudaDeviceSynchronize());

  for (auto i = 0; i < 10; ++i) {
    printf("\n");
    printf("prefixN[%d] = %d\n", i, radix_data.prefixN[i]);
    printf("hasLeafLeft[%d] = %d\n", i, radix_data.hasLeafLeft[i]);
    printf("hasLeafRight[%d] = %d\n", i, radix_data.hasLeafRight[i]);
    printf("leftChild[%d] = %d\n", i, radix_data.leftChild[i]);
    printf("parent[%d] = %d\n", i, radix_data.parent[i]);
  }

  // ------ Verify the result --------------------------------------------------

  RadixTreeData radix_data_host;
  radix_data_host.n_nodes = radix_data.n_nodes;
  radix_data_host.prefixN = new uint8_t[num_unique];
  radix_data_host.hasLeafLeft = new bool[num_unique];
  radix_data_host.hasLeafRight = new bool[num_unique];
  radix_data_host.leftChild = new int[num_unique];
  radix_data_host.parent = new int[num_unique];

  k_BuildRadixTree(radix_data_host.n_nodes,
                   u_sort,
                   radix_data_host.prefixN,
                   radix_data_host.hasLeafLeft,
                   radix_data_host.hasLeafRight,
                   radix_data_host.leftChild,
                   radix_data_host.parent);

  for (auto i = 0; i < radix_data_host.n_nodes; ++i) {
    if (radix_data.prefixN[i] != radix_data_host.prefixN[i]) {
      spdlog::error("prefixN[{}] = {} != {}",
                    i,
                    radix_data.prefixN[i],
                    radix_data_host.prefixN[i]);
    }
    if (radix_data.hasLeafLeft[i] != radix_data_host.hasLeafLeft[i]) {
      spdlog::error("hasLeafLeft[{}] = {} != {}",
                    i,
                    radix_data.hasLeafLeft[i],
                    radix_data_host.hasLeafLeft[i]);
    }
    if (radix_data.hasLeafRight[i] != radix_data_host.hasLeafRight[i]) {
      spdlog::error("hasLeafRight[{}] = {} != {}",
                    i,
                    radix_data.hasLeafRight[i],
                    radix_data_host.hasLeafRight[i]);
    }
    if (radix_data.leftChild[i] != radix_data_host.leftChild[i]) {
      spdlog::error("leftChild[{}] = {} != {}",
                    i,
                    radix_data.leftChild[i],
                    radix_data_host.leftChild[i]);
    }
    if (radix_data.parent[i] != radix_data_host.parent[i]) {
      spdlog::error("parent[{}] = {} != {}",
                    i,
                    radix_data.parent[i],
                    radix_data_host.parent[i]);
    }
  }

  free(radix_data_host.prefixN);
  free(radix_data_host.hasLeafLeft);
  free(radix_data_host.hasLeafRight);
  free(radix_data_host.leftChild);
  free(radix_data_host.parent);

  // ---------------------------------------------------------------------------

  checkCudaErrors(cudaFree(u_sort));
  checkCudaErrors(cudaFree(radix_data.prefixN));
  checkCudaErrors(cudaFree(radix_data.hasLeafLeft));
  checkCudaErrors(cudaFree(radix_data.hasLeafRight));
  checkCudaErrors(cudaFree(radix_data.leftChild));
  checkCudaErrors(cudaFree(radix_data.parent));

  return 0;
}