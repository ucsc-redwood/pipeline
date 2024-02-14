#pragma once

#include <benchmark/benchmark.h>

#include <algorithm>
#include <glm/glm.hpp>
#include <iostream>
#include <random>

namespace bm = benchmark;

#include "config.cuh"
#include "cuda/kernels/all.cuh"
#include "cuda_bench_helper.cuh"
#include "execution.hpp"
#include "shared/morton.h"
#include "shared/types.h"

class GpuFixture : public bm::Fixture {
 public:
  void SetUp(bm::State& st) override {
    u_points = AllocateManaged<glm::vec4>(kN);

    std::mt19937 gen(kRandomSeed);
    std::uniform_real_distribution<float> dis(kMin, kMax);
    std::generate(EXE_PAR, u_points, u_points + kN, [&]() {
      return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
    });

    u_sort = AllocateManaged<unsigned int>(kN);
    std::transform(EXE_PAR, u_points, u_points + kN, u_sort, [](const auto& p) {
      return shared::xyz_to_morton32(p, kMin, kMax);
    });
    std::sort(EXE_PAR, u_sort, u_sort + kN);

    auto it = std::unique(u_sort, u_sort + kN);
    num_unique = std::distance(u_sort, it);

    std::cout << "num_unique: " << num_unique << std::endl;

    tree.n_nodes = num_unique - 1;
    tree.prefixN = AllocateManaged<uint8_t>(tree.n_nodes);
    tree.hasLeafLeft = AllocateManaged<bool>(tree.n_nodes);
    tree.hasLeafRight = AllocateManaged<bool>(tree.n_nodes);
    tree.leftChild = AllocateManaged<int>(tree.n_nodes);
    tree.parent = AllocateManaged<int>(tree.n_nodes);

    gpu::k_BuildRadixTree<<<64, 768>>>(num_unique,
                                       u_sort,
                                       tree.prefixN,
                                       tree.hasLeafLeft,
                                       tree.hasLeafRight,
                                       tree.leftChild,
                                       tree.parent);
    cudaDeviceSynchronize();
  }

  void TearDown(bm::State& st) override {
    Free(u_points);
    Free(u_sort);
    Free(tree.prefixN);
    Free(tree.hasLeafLeft);
    Free(tree.hasLeafRight);
    Free(tree.leftChild);
    Free(tree.parent);
  }

  int num_unique;
  int num_oct_nodes;

  glm::vec4* u_points;
  unsigned int* u_sort;
  RadixTreeData tree;
};