#include <benchmark/benchmark.h>

#include <algorithm>
#include <random>

// #include "bm_01_morton.cuh"
#include "cuda/kernels/01_morton.cuh"
#include "cuda_bench_helper.cuh"
namespace bm = benchmark;

constexpr size_t kN = 10'000'000;
constexpr float kMin = 0.0f;
constexpr float kMax = 1024.0f;
constexpr float kRange = kMax - kMin;
constexpr int kRandomSeed = 114514;

class GpuFixture : public bm::Fixture {
 public:
  GpuFixture() {
    u_points = AllocateManaged<glm::vec4>(kN);
    u_points_out = AllocateManaged<unsigned int>(kN);

    std::mt19937 gen(kRandomSeed);
    std::uniform_real_distribution<float> dis(kMin, kMax);
    std::generate(u_points, u_points + kN, [&]() {
      return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
    });
  }

  ~GpuFixture() {
    Free(u_points);
    Free(u_points_out);
  }

  glm::vec4* u_points;
  unsigned int* u_points_out;
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
    ->Unit(bm::kMicrosecond);

// static void BM_Morton32(bm::State& st) {
//   const auto n_blocks = st.range(0);
//   const auto block_size =
//       DetermineBlockSizeAndDisplay(gpu::k_ComputeMortonCode, st);

//   const auto u_points = AllocateManaged<glm::vec4>(kN);
//   const auto u_morton = AllocateManaged<unsigned int>(kN);

//   std::mt19937 gen(kRandomSeed);
//   std::uniform_real_distribution<float> dis(kMin, kMax);
//   std::generate(u_points, u_points + kN, [&]() {
//     return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
//   });

//   for (auto _ : st) {
//     CudaEventTimer timer(st, true);

//     gpu::k_ComputeMortonCode<<<n_blocks, block_size>>>(
//         u_points, u_morton, kN, kMin, kRange);
//   }

//   Free(u_points);
//   Free(u_morton);
// }

// BENCHMARK(BM_Morton32)
//     ->RangeMultiplier(2)
//     ->Range(1, 1 << 10)
//     ->UseManualTime()
//     ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
