#include <algorithm>

#include "kernels/impl/simple_radix_sort.hpp"

void k_SortKeysInplace(unsigned int *keys, const int n) {
  std::sort(keys, keys + n);
}

void k_SimpleRadixSort(unsigned int *keys, const int n) {
  cpu::omp_lsd_radix_sort(n, keys);
}

void k_SimpleRadixSort(unsigned int *keys,
                       unsigned int *keys_alt,
                       const int n) {
  cpu::omp_lsd_radix_sort(n, keys, keys_alt);
}
