#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>
#include <random>

// #include "bm_01_morton.cuh"
#include "cuda/kernels/01_morton.cuh"
#include "cuda_bench_helper.cuh"
#include "execution.hpp"
#include "shared/morton.h"
namespace bm = benchmark;

constexpr size_t kN = 10'000'000;
constexpr float kMin = 0.0f;
constexpr float kMax = 1024.0f;
constexpr float kRange = kMax - kMin;
constexpr int kRandomSeed = 114514;

// For Cuda, I think we want to use 'Setup' and 'TearDown'
class GpuFixture : public bm::Fixture {
 public:
  void SetUp(bm::State& st) override {
    u_points = AllocateManaged<glm::vec4>(kN);
    u_points_out = AllocateManaged<unsigned int>(kN);

    std::mt19937 gen(kRandomSeed);
    std::uniform_real_distribution<float> dis(kMin, kMax);
    std::generate(EXE_PAR, u_points, u_points + kN, [&]() {
      return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
    });

    u_sort = AllocateManaged<unsigned int>(kN);
    std::transform(EXE_PAR, u_points, u_points + kN, u_sort, [](const auto& p) {
      return shared::xyz_to_morton32(p, kMin, kMax);
    });

    u_sorted = AllocateManaged<unsigned int>(kN);
    std::copy(EXE_PAR, u_sort, u_sort + kN, u_sorted);
    std::sort(EXE_PAR, u_sorted, u_sorted + kN);

    // peek 10 sorted
    std::cout << "Sorted 10: ";
    for (int i = 0; i < 10; i++) {
      std::cout << u_sorted[i] << " ";
    }
  }

  void TearDown(bm::State& st) override {
    Free(u_points);
    Free(u_points_out);
  }

  glm::vec4* u_points;
  unsigned int* u_points_out;

  unsigned int* u_sort;

  unsigned int* u_sorted;
};

BENCHMARK_DEFINE_F(GpuFixture, BM_Morton32)(bm::State& st) {
  const auto num_blocks = st.range(0);
  const auto block_size =
      DetermineBlockSizeAndDisplay(gpu::k_ComputeMortonCode, st);

  for (auto _ : st) {
    CudaEventTimer timer(st, true);

    gpu::k_ComputeMortonCode<<<num_blocks, block_size>>>(
        u_points, u_points_out, kN, kMin, kRange);
  }
}

BENCHMARK_REGISTER_F(GpuFixture, BM_Morton32)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 10)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

int main(int argc, char** argv) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

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
