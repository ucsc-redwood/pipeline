#include <benchmark/benchmark.h>

#include "bm_01_morton.cuh"
#include "bm_02_sort.cuh"
#include "bm_04_radix_tree.cuh"
#include "bm_05_edge_count.cuh"
#include "bm_07_octree.cuh"

int main(int argc, char** argv) {
  int device_count;
  cudaGetDeviceCount(&device_count);

  cudaSetDevice(0);

  for (auto device_id = 0; device_id < device_count; ++device_id) {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);

    std::cout << "Device ID: " << device_id << '\n';
    std::cout << "Device name: " << device_prop.name << '\n';
    std::cout << "Compute capability: " << device_prop.major << "."
              << device_prop.minor << '\n';
    std::cout << "Total global memory: "
              << device_prop.totalGlobalMem / (1024 * 1024) << " MB" << '\n';
    std::cout << "Number of multiprocessors: "
              << device_prop.multiProcessorCount << '\n';
    std::cout << "Max threads per block: " << device_prop.maxThreadsPerBlock
              << '\n';
    std::cout << "Max threads per multiprocessor: "
              << device_prop.maxThreadsPerMultiProcessor << '\n';
    std::cout << "Warp size: " << device_prop.warpSize << '\n';
    std::cout << '\n';
  }

  bm::Initialize(&argc, argv);
  bm::RunSpecifiedBenchmarks();
}
