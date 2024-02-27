#pragma once

// Fixed for 4 passes, 256 radix
// This is what we care, no need to generalize it
template <typename T>
constexpr void MallocManaged(T** ptr, const size_t num_items) {
  cudaMallocManaged(ptr, num_items * sizeof(T));
}

#define ATTACH_STREAM_SINGLE(ptr) \
  cudaStreamAttachMemAsync(stream, ptr, 0, cudaMemAttachSingle)

// Hard coded for 4 passes, 256 threads
struct OneSweep {
  explicit OneSweep(const int n) : n(n) {
    MallocManaged(&u_sort, n);
    MallocManaged(&u_sort_alt, n);
    MallocManaged(&u_global_histogram, 256 * 4);
    MallocManaged(&u_index, 4);
    for (auto& u_pass_histogram : u_pass_histograms) {
      MallocManaged(&u_pass_histogram, 256 * binningThreadblocks(n));
    }
  }

  ~OneSweep() {
    cudaFree(u_sort);
    cudaFree(u_sort_alt);
    cudaFree(u_global_histogram);
    cudaFree(u_index);
    for (const auto& u_pass_histogram : u_pass_histograms) {
      cudaFree(u_pass_histogram);
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

  [[nodiscard]] size_t getMemorySize() const {
    size_t total = 0;
    total += n * sizeof(unsigned int);
    total += n * sizeof(unsigned int);
    total += 256 * 4 * sizeof(unsigned int);
    total += 4 * sizeof(unsigned int);
    total += 4 * 256 * binningThreadblocks(n) * sizeof(unsigned int);
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
