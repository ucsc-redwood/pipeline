#include <benchmark/benchmark.h>
#include <omp.h>

#include <execution>
#include <glm/glm.hpp>

#include "config.hpp"
#include "kernels/morton.hpp"

namespace bm = benchmark;

static void BM_SinglePointToCode(bm::State& st) {
  auto x = 0.0f;
  auto y = 0.0f;
  auto z = 0.0f;

  for (auto _ : st) {
    auto ret = cpu::single_point_to_code_v2(x, y, z, kMin, kRange);
    bm::DoNotOptimize(ret);
  }
}

static void BM_Morton32(bm::State& st) {
  const auto num_threads = st.range(0);

  auto data = new glm::vec4[kN];
  auto morton_keys = new unsigned int[kN];

  omp_set_num_threads(num_threads);

  for (auto _ : st) {
    k_ComputeMortonCode(data, morton_keys, kN, kMin, kRange);
  }

  delete[] data;
  delete[] morton_keys;
}

static void BM_StdTransform(bm::State& st) {
  auto data = new glm::vec4[kN];
  auto morton_keys = new unsigned int[kN];

  for (auto _ : st) {
    std::transform(std::execution::par_unseq,
                   data,
                   data + kN,
                   morton_keys,
                   [&](const auto& v) {
                     return cpu::single_point_to_code_v2(
                         v.x, v.y, v.z, kMin, kRange);
                   });
  }

  delete[] data;
  delete[] morton_keys;
}

BENCHMARK(BM_SinglePointToCode);

BENCHMARK(BM_StdTransform)->Unit(bm::kMillisecond);

BENCHMARK(BM_Morton32)
    ->RangeMultiplier(2)
    ->Range(1, 48)
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
