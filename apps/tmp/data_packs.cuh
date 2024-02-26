#pragma once

// Fixed for 4 passes, 256 radix
// This is what we care, no need to generalize it
namespace {

template <typename T>
constexpr void MallocManaged(T** ptr, size_t num_items) {
  cudaMallocManaged(ptr, num_items * sizeof(T));
}

#define AttachStreamSingle(ptr) \
  cudaStreamAttachMemAsync(stream, ptr, 0, cudaMemAttachSingle)

}  // namespace

class OneSweep {
 public:
  explicit OneSweep(const int n) : n(n) {
    MallocManaged(&u_sort, n);
    MallocManaged(&u_sort_alt, n);
    MallocManaged(&u_global_histogram, 256 * 4);
    MallocManaged(&u_index, 4);
    for (auto i = 0; i < 4; i++) {
      MallocManaged(&u_pass_histograms[i], 256 * binningThreadblocks(n));
    }
  }

  ~OneSweep() {
    cudaFree(u_sort);
    cudaFree(u_sort_alt);
    cudaFree(u_global_histogram);
    cudaFree(u_index);
    for (auto i = 0; i < 4; i++) {
      cudaFree(u_pass_histograms[i]);
    }
  }

  [[nodiscard]] unsigned int* getSort() { return u_sort; }
  [[nodiscard]] const unsigned int* getSort() const { return u_sort; }

  void attachStream(cudaStream_t& stream) {
    AttachStreamSingle(u_sort);
    AttachStreamSingle(u_sort_alt);
    AttachStreamSingle(u_global_histogram);
    AttachStreamSingle(u_index);
    for (auto i = 0; i < 4; i++) {
      AttachStreamSingle(u_pass_histograms[i]);
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

 protected:
  int n;

 private:
  unsigned int* u_sort;
  unsigned int* u_sort_alt;
  unsigned int* u_global_histogram;
  unsigned int* u_index;
  unsigned int* u_pass_histograms[4];

  friend void Dispatch_SortKernels(OneSweep& one_sweep,
                                   int n,
                                   int grid_size,
                                   const cudaStream_t& stream);
};
