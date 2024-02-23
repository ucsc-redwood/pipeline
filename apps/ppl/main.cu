#include <omp.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>

#include "app_params.hpp"
#include "gpu_kernels.cuh"
#include "kernels/all.hpp"
#include "pipe.cuh"
#include "shared/types.h"

namespace {

void k_StdSort(unsigned int* u_data, const int n) {
  std::sort(u_data, u_data + n);
}

}  // namespace

int main(const int argc, const char* argv[]) {
  constexpr auto n = 1920 * 1080;  // 2.0736M
  int n_threads = 4;
  int my_num_blocks = 64;
  bool debug_print = false;

  CLI::App app{"Multi-threaded sorting benchmark"};

  app.add_option("-t,--threads", n_threads, "Number of threads to use")
      ->check(CLI::Range(1, 48));

  app.add_option("-b,--blocks", my_num_blocks, "Number of blocks to use")
      ->check(CLI::PositiveNumber);

  app.add_flag("-d,--debug", debug_print, "Print debug information");

  CLI11_PARSE(app, argc, argv)

  spdlog::info("n = {}", n);
  spdlog::info("n_threads = {}", n_threads);
  spdlog::info("my_num_blocks = {}", my_num_blocks);

  // set log level to debug
  if (debug_print) {
    spdlog::set_level(spdlog::level::debug);
  }

  omp_set_num_threads(n_threads);

#pragma omp parallel
  { printf("Hello from thread %d\n", omp_get_thread_num()); }

  AppParams params{
      .n = n,
      .min = 0.0f,
      .max = 1024.0f,
      .range = 1024.0f,
      .seed = 114514,
      .my_num_blocks = my_num_blocks,
  };

  // spdlog all params in one line
  spdlog::info(
      "params: n={}, min={}, max={}, range={}, seed={}, my_num_blocks={}",
      params.n,
      params.min,
      params.max,
      params.range,
      params.seed,
      params.my_num_blocks);

  glm::vec4* u_points;
  cudaMallocManaged(&u_points, n * sizeof(glm::vec4));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto one_sweep = gpu::OneSweepHelper::CreateOnesweepData(n);
  gpu::OneSweepHelper::AttachOnesweepStream(one_sweep, stream);

  int* num_unique_out;
  cudaMallocManaged(&num_unique_out, sizeof(int));

  // -----------------------------------------------------------------------

  gpu::Disptach_InitRandomVec4(u_points, params, params.my_num_blocks, stream);

  gpu::Dispatch_MortonCompute(
      u_points, one_sweep.u_sort, params, params.my_num_blocks, stream);

  gpu::Dispatch_SortKernels(one_sweep, n, params.my_num_blocks, stream);

  gpu::Dispatch_CountUnique(
      one_sweep.u_sort, num_unique_out, n, params.my_num_blocks, stream);

  cudaDeviceSynchronize();

  // -----------------------------------------------------------------------

  auto is_sorted = std::is_sorted(one_sweep.u_sort, one_sweep.u_sort + n);

  spdlog::info("is_sorted = {}", is_sorted);

  cudaFree(u_points);
  gpu::OneSweepHelper::DestroyOnesweepData(one_sweep);

  cudaStreamDestroy(stream);

  return 0;
}