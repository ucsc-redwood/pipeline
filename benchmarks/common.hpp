#pragma once

#include <benchmark/benchmark.h>
#include <omp.h>

#include <glm/glm.hpp>

namespace bm = benchmark;

#include "config.hpp"
#include "kernels/all.hpp"

template <typename T>
[[nodiscard]] T* AllocateHost(const size_t n_items) {
  return static_cast<T*>(malloc(n_items * sizeof(T)));
}

inline void Free(void* ptr) { free(ptr); }

class MyFixture : public bm::Fixture {
 public:
  MyFixture() {
    u_input = AllocateHost<glm::vec4>(kN);
    u_input_out = AllocateHost<unsigned int>(kN);
    u_morton = AllocateHost<unsigned int>(kN);
  }

  ~MyFixture() {
    Free(u_input);
    Free(u_input_out);
    Free(u_morton);
  }

  //   void SetUp(const bm::State& state) override {}

  //   void TearDown(const bm::State& state) override {}

  glm::vec4* u_input;
  unsigned int* u_input_out;
  unsigned int* u_morton;
};
