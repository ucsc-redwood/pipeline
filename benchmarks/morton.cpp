#include "kernels/morton.hpp"

#include <benchmark/benchmark.h>
#include <omp.h>

#include <glm/glm.hpp>

namespace bm = benchmark;

constexpr auto kN = 10'000'000;
constexpr auto kMin = 0.0f;
constexpr auto kMax = 1024.0f;
constexpr auto kRange = kMax - kMin;

static void BM_SinglePointToCode(bm::State& st) {
  auto x = 0.0f;
  auto y = 0.0f;
  auto z = 0.0f;

  for (auto _ : st) {
    auto ret = single_point_to_code_v2(x, y, z, kMin, kRange);
    bm::DoNotOptimize(ret);
  }
}

static void GenerateInput(glm::vec4* data, float min, float max) {
  srand(114514);
  const float range = max - min;

#pragma omp parallel for schedule(static)
  for (auto i = 0; i < kN; i++) {
    data[i][0] = (float)rand() / (float)RAND_MAX * range + min;
    data[i][1] = (float)rand() / (float)RAND_MAX * range + min;
    data[i][2] = (float)rand() / (float)RAND_MAX * range + min;
    data[i][3] = 1.0f;
  }
}

static void BM_Morton32(bm::State& st) {
  const auto num_threads = st.range(0);

  // use malloc to allocate memory
  auto data = new glm::vec4[kN];
  auto morton_keys = new unsigned int[kN];

  GenerateInput(data, kMin, kMax);

  omp_set_num_threads(num_threads);
  for (auto _ : st) {
    convert_xyz_to_morton_code(data, morton_keys, kN, kMin, kRange);
    bm::DoNotOptimize(morton_keys[0]);
  }

  delete[] data;
  delete[] morton_keys;
}

BENCHMARK(BM_SinglePointToCode);

BENCHMARK(BM_Morton32)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
