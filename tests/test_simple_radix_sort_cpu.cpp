
#include <omp.h>

#include <algorithm>
#include <iostream>

#include "kernels/02_sort.hpp"

int main(const int argc, const char** argv) {
  int n = argc - 1;
  auto* data = new unsigned int[n];
  for (int i = 0; i < n; i++) {
    data[i] = atoi(argv[i + 1]);
  }

  omp_set_num_threads(4);
  k_SimpleRadixSort(data, n);

  for (int i = 0; i < n; i++) std::cout << data[i] << " ";

  delete[] data;
  return 0;
}
