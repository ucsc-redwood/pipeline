#include <benchmark/benchmark.h>

#include "bm_01_morton.cuh"
#include "bm_02_sort.cuh"

int main(int argc, char** argv) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  cudaSetDevice(0);

  for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    std::cout << "Device ID: " << deviceId << std::endl;
    std::cout << "Device name: " << deviceProp.name << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "."
              << deviceProp.minor << std::endl;
    std::cout << "Total global memory: "
              << deviceProp.totalGlobalMem / (1024 * 1024) << " MB"
              << std::endl;
    std::cout << "Number of multiprocessors: " << deviceProp.multiProcessorCount
              << std::endl;
    std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock
              << std::endl;
    std::cout << "Max threads per multiprocessor: "
              << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Warp size: " << deviceProp.warpSize << std::endl;
    std::cout << std::endl;
  }

  bm::Initialize(&argc, argv);
  bm::RunSpecifiedBenchmarks();
}
