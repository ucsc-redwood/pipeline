#include <omp.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>

#include "app_params.hpp"
#include "better_oct.cuh"
#include "cuda/common/helper_cuda.hpp"
#include "gpu_kernels.cuh"
#include "shared/morton.h"
#include "shared/oct_v2.h"

struct RadixTree {
  explicit RadixTree(const int n) : n_nodes(n) {
    MallocManaged(&prefixN, n);
    MallocManaged(&hasLeafLeft, n);
    MallocManaged(&hasLeafRight, n);
    MallocManaged(&leftChild, n);
    MallocManaged(&parent, n);
  }

  ~RadixTree() {
    cudaFree(prefixN);
    cudaFree(hasLeafLeft);
    cudaFree(hasLeafRight);
    cudaFree(leftChild);
    cudaFree(parent);
  }

  void attachStream(cudaStream_t& stream) {
    AttachStreamSingle(prefixN);
    AttachStreamSingle(hasLeafLeft);
    AttachStreamSingle(hasLeafRight);
    AttachStreamSingle(leftChild);
    AttachStreamSingle(parent);
  }

  [[nodiscard]] size_t getMemorySize() const {
    size_t total = 0;
    total += n_nodes * sizeof(uint8_t);
    total += n_nodes * sizeof(bool);
    total += n_nodes * sizeof(bool);
    total += n_nodes * sizeof(int);
    total += n_nodes * sizeof(int);
    return total;
  }

  [[nodiscard]] int getNumNodes() const { return n_nodes; }

  const int n_nodes;
  uint8_t* prefixN;
  bool* hasLeafLeft;
  bool* hasLeafRight;
  int* leftChild;
  int* parent;
};

// We need to assume n != #unique
struct Pipe {
  explicit Pipe(const int n) : n(n), one_sweep(n), brt(n), oct(0.6 * n) {
    MallocManaged(&u_points, n);
    MallocManaged(&u_edge_count, n);
    MallocManaged(&u_prefix_sum, n);
    MallocManaged(&u_num_unique, 1);
    MallocManaged(&num_oct_nodes_out, 1);
  }

  ~Pipe() {
    cudaFree(u_points);
    cudaFree(u_edge_count);
    cudaFree(u_prefix_sum);
    cudaFree(u_num_unique);
    cudaFree(num_oct_nodes_out);
  }

  void attachStream(cudaStream_t& stream) {
    one_sweep.attachStream(stream);
    brt.attachStream(stream);
    oct.attachStream(stream);
    AttachStreamSingle(u_points);
    AttachStreamSingle(u_edge_count);
    AttachStreamSingle(u_prefix_sum);
    AttachStreamSingle(u_num_unique);
    AttachStreamSingle(num_oct_nodes_out);
  }

  [[nodiscard]] size_t getMemorySize() const {
    size_t total = 0;
    total += n * sizeof(glm::vec4);
    total += n * sizeof(int);
    total += n * sizeof(int);
    total += sizeof(int);
    total += one_sweep.getMemorySize();
    total += brt.getMemorySize();
    total += oct.getMemorySize();
    return total;
  }

  [[nodiscard]] int getNumPoints() const { return n; }
  [[nodiscard]] int getNumUnique_unsafe() const { return *u_num_unique; }
  [[nodiscard]] int getNumBrtNodes_unsafe() const {
    return getNumUnique_unsafe() - 1;
  }
  [[nodiscard]] const unsigned int* getMortonKeys() const {
    return one_sweep.u_sort;
  }

  glm::vec4* u_points;
  OneSweep one_sweep;
  RadixTree brt;
  int* u_edge_count;
  int* u_prefix_sum;
  OctNodes_better oct;

  const int n;
  // unfortunately, we need to use a pointer here, because these values depend
  // on the computation resutls
  int* u_num_unique;
  int* num_oct_nodes_out;
};

// write a ostream function for glm::vec4
std::ostream& operator<<(std::ostream& os, const glm::vec4& v) {
  os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
  return os;
}

int main(const int argc, const char* argv[]) {
  constexpr auto n = 1920 * 1080;  // 2.0736M
  constexpr auto educated_guess_n_oct_nodes = 0.6;
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

  auto pipe = std::make_unique<Pipe>(n);
  pipe->attachStream(stream);

  const auto mem_size = pipe->getMemorySize();
  spdlog::info("Pipe Memory size: {} MB", mem_size / 1024 / 1024);

  // -----------------------------------------------------------------------

  gpu::Disptach_InitRandomVec4(
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
    if (safety_ratio > 100 * educated_guess_n_oct_nodes) {
      spdlog::error("pre allocated num_oct_nodes is too small");
    }
  }

  // -----------------------------------------------------------------------
  // tmp

  gpu::v2::k_MakeOctNodes_Deps<<<params.my_num_blocks, 768, 0, stream>>>(
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

  gpu::v2::k_LinkLeafNodes_Deps<<<params.my_num_blocks, 768, 0, stream>>>(
      pipe->oct.u_children,
      pipe->oct.u_corner,
      pipe->oct.u_cell_size,
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

  // peek 10 oct nodes, cornor, cell_size, parent, child_node_mask
  for (int i = 0; i < 10; i++) {
    std::cout << "oct_nodes[" << i << "]:\n";
    std::cout << "\tcorner: " << pipe->oct.u_corner[i] << "\n";
    std::cout << "\tcell_size: " << pipe->oct.u_cell_size[i] << "\n";
    std::cout << "\tparent: " << pipe->brt.parent[i] << "\n";
    std::cout << "\tchild_node_mask: " << pipe->oct.u_child_node_mask[i]
              << "\n\n";
  }

  checkCudaErrors(cudaStreamDestroy(stream));
  return 0;
}
