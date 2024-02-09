#include <iostream>
#include "../../benchmarks/common.cuh"

int main(const int argc, const char** argv) {
  int n = 10'000'000;
  auto [d_sort, d_tree] = gpu::MakeRadixTree_Fake();

  

  cudaFree(d_sort);
  cudaFree(d_tree.prefixN);
  cudaFree(d_tree.hasLeafLeft);
  cudaFree(d_tree.hasLeafRight);
  cudaFree(d_tree.leftChild);
  cudaFree(d_tree.parent);
  return 0;
}