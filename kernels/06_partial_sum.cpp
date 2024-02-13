#include "kernels/06_partial_sum.hpp"

int* k_PartialSum(const int* data,
                  std::ptrdiff_t first,
                  const std::ptrdiff_t last,
                  int* d_first) {
  if (first == last) return d_first;
  auto sum = data[first];
  *d_first = sum;
  while (++first != last) {
    sum += data[first];
    *++d_first = sum;
  }
  return ++d_first;
}
