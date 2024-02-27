#pragma once

#include <glm/glm.hpp>

#include "brt.cuh"
#include "oct.cuh"
#include "one_sweep.cuh"

// Note: We need to assume n != #unique
//
struct Pipe {
  explicit Pipe(const int n) : one_sweep(n), brt(n), oct(0.6 * n), n(n) {
    MallocManaged(&u_points, n);
    MallocManaged(&u_edge_count, n);
    MallocManaged(&u_prefix_sum, n);
    MallocManaged(&u_num_unique, 1);
    MallocManaged(&num_oct_nodes_out, 1);
  }

  Pipe(const Pipe&) = delete;
  Pipe& operator=(const Pipe&) = delete;
  Pipe(Pipe&&) = delete;
  Pipe& operator=(Pipe&&) = delete;

  ~Pipe() {
    cudaFree(u_points);
    cudaFree(u_edge_count);
    cudaFree(u_prefix_sum);
    cudaFree(u_num_unique);
    cudaFree(num_oct_nodes_out);
  }

  void attachStream(const cudaStream_t stream) {
    one_sweep.attachStream(stream);
    brt.attachStream(stream);
    oct.attachStream(stream);
    ATTACH_STREAM_SINGLE(u_points);
    ATTACH_STREAM_SINGLE(u_edge_count);
    ATTACH_STREAM_SINGLE(u_prefix_sum);
    ATTACH_STREAM_SINGLE(u_num_unique);
    ATTACH_STREAM_SINGLE(num_oct_nodes_out);
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
    return one_sweep.getSort();
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