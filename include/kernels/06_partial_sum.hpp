#pragma once

#include <cstddef>

// https://en.cppreference.com/w/cpp/algorithm/partial_sum#Version_1
[[nodiscard]] int* k_PartialSum(const int* data,
                                std::ptrdiff_t first,
                                std::ptrdiff_t last,
                                int* d_first);
