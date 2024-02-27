#pragma once

#include "helper.cuh"

// Fixed for 4 passes, 256 radix
// This is what we care, no need to generalize it
//
struct OneSweep {
  static constexpr auto kRadix = 256;

  explicit OneSweep(const int n) : n(n) {
    MallocManaged(&u_sort, n);
    MallocManaged(&u_sort_alt, n);
    MallocManaged(&u_global_histogram, kRadix * 4);
    MallocManaged(&u_index, 4);
    for (auto& u_pass_histogram : u_pass_histograms) {
      MallocManaged(&u_pass_histogram, kRadix * binningThreadblocks(n));
    }
  }

  OneSweep(const OneSweep&) = delete;
  OneSweep& operator=(const OneSweep&) = delete;
  OneSweep(OneSweep&&) = delete;
  OneSweep& operator=(OneSweep&&) = delete;

  ~OneSweep() {
    CUDA_FREE(u_sort);
    CUDA_FREE(u_sort_alt);
    CUDA_FREE(u_global_histogram);
    CUDA_FREE(u_index);
    for (const auto& u_pass_histogram : u_pass_histograms) {
      CUDA_FREE(u_pass_histogram);
    }
  }

  [[nodiscard]] unsigned int* getSort() { return u_sort; }
  [[nodiscard]] const unsigned int* getSort() const { return u_sort; }

  void attachStream(const cudaStream_t stream) const {
    ATTACH_STREAM_SINGLE(u_sort);
    ATTACH_STREAM_SINGLE(u_sort_alt);
    ATTACH_STREAM_SINGLE(u_global_histogram);
    ATTACH_STREAM_SINGLE(u_index);
    for (const auto u_pass_histogram : u_pass_histograms) {
      ATTACH_STREAM_SINGLE(u_pass_histogram);
    }
  }

  // for ~2m nodes, its about 16MB
  [[nodiscard]] size_t getMemorySize() const {
    size_t total = 0;
    total += n * sizeof(unsigned int);
    total += n * sizeof(unsigned int);
    total += kRadix * 4 * sizeof(unsigned int);
    total += 4 * sizeof(unsigned int);
    total += 4 * kRadix * binningThreadblocks(n) * sizeof(unsigned int);
    return total;
  }

  static constexpr int binningThreadblocks(const int size) {
    // Need to match to the numbers in the '.cu' file in gpu source code
    constexpr int partitionSize = 7680;
    return (size + partitionSize - 1) / partitionSize;
  }

  static constexpr int globalHistThreadblocks(const int size) {
    // Need to match to the numbers in the '.cu' file in gpu source code
    constexpr int globalHistPartitionSize = 65536;
    return (size + globalHistPartitionSize - 1) / globalHistPartitionSize;
  }

  const int n;
  unsigned int* u_sort;
  unsigned int* u_sort_alt;
  unsigned int* u_global_histogram;
  unsigned int* u_index;
  unsigned int* u_pass_histograms[4];
};
