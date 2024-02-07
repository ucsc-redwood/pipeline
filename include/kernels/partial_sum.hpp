#pragma once

#include <cstddef>

// https://en.cppreference.com/w/cpp/algorithm/partial_sum#Version_1
int* k_PartialSum(int* input,
                  std::ptrdiff_t first,
                  const std::ptrdiff_t last,
                  int* d_first);
