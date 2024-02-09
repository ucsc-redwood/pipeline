#include <benchmark/benchmark.h>

#include <glm/glm.hpp>

#include "config.hpp"
#include "cuda/kernels/edge_count.cuh"
#include "cuda/kernels/radix_tree.cuh"
#include "cuda_bench_helper.cuh"
#include "kernels/init.hpp"
#include "types/brt.hpp"

// auto [d_sort, d_tree] = gpu::MakeRadixTree_Fake();

namespace bm = benchmark;

BENCHMARK_MAIN();
