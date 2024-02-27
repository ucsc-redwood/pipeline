#include <cuda_runtime_api.h>
#include <omp.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <iostream>

#include "app_params.hpp"
#include "better_types/pipe.cuh"
#include "cuda/common/helper_cuda.hpp"
#include "gpu_kernels.cuh"
#include "shared/morton.h"

constexpr auto kEducatedGuessNumOctNodes = 0.6;

[[nodiscard]] std::ostream& operator<<(std::ostream& os, const glm::vec4& v) {
  os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
  return os;
}

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

#ifdef NDEBUG
  spdlog::debug("NDEBUG is defined");
#else
  spdlog::debug("NDEBUG is not defined");
#endif

  omp_set_num_threads(n_threads);

#pragma omp parallel
  { printf("Hello from thread %d\n", omp_get_thread_num()); }

  AppParams params;
  params.n = n;
  params.min = 0.0f;
  params.max = 1024.0f;
  params.range = params.max - params.min;
  params.seed = 114514;
  params.my_num_blocks = my_num_blocks;

  spdlog::info(
      "params: n={}, min={}, max={}, range={}, seed={}, my_num_blocks={}",
      params.n,
      params.min,
      params.max,
      params.range,
      params.seed,
      params.my_num_blocks);

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));

  const auto pipe = std::make_unique<Pipe>(n);
  pipe->attachStream(stream);

  const auto mem_size = pipe->getMemorySize();
  spdlog::info("Pipe Memory size: {} MB", mem_size / 1024 / 1024);

  // -----------------------------------------------------------------------

  gpu::Dispatch_InitRandomVec4(
      pipe->u_points, params, params.my_num_blocks, stream);

  gpu::Dispatch_MortonCompute(pipe->u_points,
                              pipe->one_sweep.getSort(),
                              params,
                              params.my_num_blocks,
                              stream);

  gpu::Dispatch_SortKernels(pipe->one_sweep, n, params.my_num_blocks, stream);

  gpu::Dispatch_CountUnique(pipe->one_sweep.getSort(),
                            pipe->u_num_unique,
                            n,
                            params.my_num_blocks,
                            stream);

  gpu::Dispatch_BuildRadixTree(pipe->u_num_unique,
                               pipe->one_sweep.getSort(),
                               pipe->brt.prefixN,
                               pipe->brt.hasLeafLeft,
                               pipe->brt.hasLeafRight,
                               pipe->brt.leftChild,
                               pipe->brt.parent,
                               params.my_num_blocks,
                               stream);

  gpu::Dispatch_EdgeCount(pipe->brt.prefixN,
                          pipe->brt.parent,
                          pipe->u_edge_count,
                          pipe->u_num_unique,
                          params.my_num_blocks,
                          stream);

  checkCudaErrors(cudaStreamSynchronize(stream));

  // TMP
  std::inclusive_scan(pipe->u_edge_count,
                      pipe->u_edge_count + pipe->getNumUnique_unsafe(),
                      pipe->u_prefix_sum);

  gpu::Dispatch_MakeOctree(params.my_num_blocks,
                           stream,
                           pipe->oct.u_children,
                           pipe->oct.u_corner,
                           pipe->oct.u_cell_size,
                           pipe->oct.u_child_node_mask /* Node Mask */,
                           pipe->u_prefix_sum,
                           pipe->u_edge_count,
                           pipe->one_sweep.getSort(),
                           pipe->brt.prefixN,
                           pipe->brt.parent,
                           params.min,
                           params.range,
                           pipe->u_num_unique);

  gpu::Dispatch_LinkOctreeNodes(params.my_num_blocks,
                                stream,
                                pipe->oct.u_children,
                                pipe->oct.u_child_leaf_mask /* Leaf Mask */,
                                pipe->u_prefix_sum,
                                pipe->u_edge_count,
                                pipe->one_sweep.getSort(),
                                pipe->brt.hasLeafLeft,
                                pipe->brt.hasLeafRight,
                                pipe->brt.prefixN,
                                pipe->brt.parent,
                                pipe->brt.leftChild,
                                pipe->u_num_unique);

  checkCudaErrors(cudaStreamSynchronize(stream));

  // -----------------------------------------------------------------------

  spdlog::info("pipe->getNumUnique_unsafe() = {}", pipe->getNumUnique_unsafe());

  {
    // peek 10 prefix sum
    for (int i = 0; i < 10; i++) {
      spdlog::debug("u_prefix_sum[{}] = {}", i, pipe->u_prefix_sum[i]);
    }

    const auto num_oct_nodes =
        pipe->u_prefix_sum[pipe->getNumBrtNodes_unsafe()];
    spdlog::info("num_oct_nodes = {}", num_oct_nodes);

    // print percentage of oct vs n input. "(xx/yy) = x.xx%"
    const auto safety_ratio = 100.0 * num_oct_nodes / n;
    spdlog::info("({}/{}) = {:.2f}%", num_oct_nodes, n, safety_ratio);
    if (safety_ratio > 100 * kEducatedGuessNumOctNodes) {
      spdlog::error("pre allocated num_oct_nodes is too small");
    }
  }

  // peek 10 oct nodes, cornor, cell_size, parent, child_node_mask
  for (int i = 0; i < 10; i++) {
    std::cout << "oct_nodes[" << i << "]:\n";
    std::cout << "\t corner: " << pipe->oct.u_corner[i] << "\n";
    std::cout << "\t cell_size: " << pipe->oct.u_cell_size[i] << "\n";
    std::cout << "\t parent: " << pipe->brt.parent[i] << "\n";
    std::cout << "\t child_node_mask: " << pipe->oct.u_child_node_mask[i]
              << "\n\n";
  }

  checkCudaErrors(cudaStreamDestroy(stream));
  return 0;
}
