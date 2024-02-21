#include <omp.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>

#include "cuda/kernels/all.cuh"
#include "kernels/all.hpp"
#include "pipe.cuh"
#include "shared/types.h"

namespace {

struct AppParams {
  int n;
  float min, max, range;
  int seed;
  int my_num_blocks;
};

void k_StdSort(unsigned int* u_data, const int n) {
  std::sort(u_data, u_data + n);
}

}  // namespace

// Baseline CPU implementation
void run_cpu_pass(Pipe* pipe, AppParams& params) {
  ++params.seed;
  k_InitRandomVec4(
      pipe->u_points, params.n, params.min, params.range, params.seed);

  k_ComputeMortonCode(pipe->u_points,
                      pipe->one_sweep.u_sort,
                      params.n,
                      params.min,
                      params.range);

  k_SimpleRadixSort(pipe->one_sweep.u_sort, params.n);

  k_Unique(&pipe->n_unique, pipe->one_sweep.u_sort, params.n);
}

// Baseline GPU implementation
void run_gpu_pass(Pipe* pipe, AppParams& params, const cudaStream_t& stream) {
  ++params.seed;

  // Init
  {
    constexpr auto num_threads = 768;
    constexpr auto seed = 114514;
    const auto grid_size = (params.n + num_threads - 1) / num_threads;
    gpu::k_InitRandomVec4<<<grid_size, num_threads, 0, stream>>>(
        pipe->u_points, params.n, params.min, params.range, params.seed);
  }

  // ---------------------------------------------------------------------------
  // Morton Code

  {
    constexpr auto num_threads = 768;

    gpu::k_ComputeMortonCode<<<params.my_num_blocks, num_threads, 0, stream>>>(
        pipe->u_points,
        pipe->one_sweep.u_sort,
        params.n,
        params.min,
        params.range);
  }
}

int main(const int argc, const char* argv[]) {
  constexpr auto n = 1920 * 1080;  // 2.0736M
  int n_threads = 4;
  int my_num_blocks = 64;

  CLI::App app{"Multi-threaded sorting benchmark"};

  app.add_option("-t,--threads", n_threads, "Number of threads to use")
      ->check(CLI::Range(1, 48));

  app.add_option("-b,--blocks", my_num_blocks, "Number of blocks to use")
      ->check(CLI::PositiveNumber);

  CLI11_PARSE(app, argc, argv)

  spdlog::info("n = {}", n);
  spdlog::info("n_threads = {}", n_threads);
  spdlog::info("my_num_blocks = {}", my_num_blocks);

  omp_set_num_threads(n_threads);

  AppParams params{
      .n = n,
      .min = 0.0f,
      .max = 1024.0f,
      .range = 1024.0f,
      .seed = 114514,
      .my_num_blocks = my_num_blocks,
  };

  constexpr auto n_frames = 100;

  constexpr auto n_streams = 2;

  std::array<cudaStream_t, n_streams> streams;
  for (auto i = 0; i < n_streams; ++i) {
    checkCudaErrors(cudaStreamCreate(&streams[i]));
  }

  // error: no default constructor exists for class "Pipe"
  // std::unique_ptr<Pipe[]> pipes(new Pipe[n_streams]{n});
  // std::array<Pipe, n_streams> pipes{n};

  std::array<Pipe, n_streams> pipes{Pipe(n), Pipe(n)};

  // cudaStreamAttachMemAsync

  for (auto i = 0; i < n_streams; ++i) {
    checkCudaErrors(cudaStreamAttachMemAsync(
        streams[i], pipes[i].u_points, 0, cudaMemAttachSingle));
    checkCudaErrors(cudaStreamAttachMemAsync(
        streams[i], pipes[i].one_sweep.u_sort, 0, cudaMemAttachSingle));
    checkCudaErrors(cudaStreamAttachMemAsync(
        streams[i], pipes[i].one_sweep.u_sort_alt, 0, cudaMemAttachSingle));
    checkCudaErrors(
        cudaStreamAttachMemAsync(streams[i],
                                 pipes[i].one_sweep.u_global_histogram,
                                 0,
                                 cudaMemAttachSingle));
    checkCudaErrors(cudaStreamAttachMemAsync(
        streams[i], pipes[i].one_sweep.u_index, 0, cudaMemAttachSingle));
    for (int j = 0; j < 4; j++) {
      checkCudaErrors(
          cudaStreamAttachMemAsync(streams[i],
                                   pipes[i].one_sweep.u_pass_histograms[j],
                                   0,
                                   cudaMemAttachSingle));
    }
  }

  // need to compute the frame rate of this loop
  auto start = std::chrono::steady_clock::now();

  for (auto i = 0; i < n_frames; ++i) {
    run_gpu_pass(&pipes[i % n_streams], params, streams[i % n_streams]);
    spdlog::info("[{}/{}] ", i, n_frames);
  }
  checkCudaErrors(cudaDeviceSynchronize());

  auto end = std::chrono::steady_clock::now();

  auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  double fps = static_cast<double>(n_frames) / (elapsed_ms / 1000.0);

  spdlog::info("Elapsed time: {} ms", elapsed_ms);
  spdlog::info("FPS: {}", fps);

  for (auto i = 0; i < n_streams; ++i) {
    checkCudaErrors(cudaStreamDestroy(streams[i]));
  }

  return 0;
}