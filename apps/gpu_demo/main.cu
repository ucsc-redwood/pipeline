#include <cuda_runtime.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <glm/glm.hpp>

#include "cuda/common/helper_cuda.hpp"
#include "cuda/kernels/all.cuh"

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
  spdlog::info("my_num_blocks = {}", my_num_blocks);

  omp_set_num_threads(n_threads);

#pragma omp parallel
  { printf("Hello, world! I'm thread %d\n", omp_get_thread_num()); }

  // ---------------------------------------------------------------------------

  constexpr auto min = 0.0f;
  constexpr auto max = 1024.0f;
  constexpr auto range = max - min;

  glm::vec4* u_data;
  unsigned int* u_morton_keys;
  checkCudaErrors(cudaMallocManaged(&u_data, n * sizeof(glm::vec4)));
  checkCudaErrors(cudaMallocManaged(&u_morton_keys, n * sizeof(unsigned int)));

  {
    constexpr auto num_threads = 768;
    constexpr auto seed = 114514;
    const auto grid_size = (n + num_threads - 1) / num_threads;
    gpu::k_InitRandomVec4<<<grid_size, num_threads>>>(
        u_data, n, min, range, seed);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  // peek 10 elements
  for (int i = 0; i < 10; ++i) {
    spdlog::debug("u_data[{}] = ({}, {}, {}, {})",
                  i,
                  u_data[i].x,
                  u_data[i].y,
                  u_data[i].z,
                  u_data[i].w);
  }

  {
    constexpr auto num_threads = 768;
    gpu::k_ComputeMortonCode<<<my_num_blocks, num_threads>>>(
        u_data, u_morton_keys, n, min, range);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  // peek 10 elements
  for (int i = 0; i < 10; ++i) {
    spdlog::info("u_morton_keys[{}] = {}", i, u_morton_keys[i]);
  }

  checkCudaErrors(cudaFree(u_data));
  checkCudaErrors(cudaFree(u_morton_keys));

  return 0;
}