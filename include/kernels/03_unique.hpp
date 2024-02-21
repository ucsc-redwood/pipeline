#pragma once

// CPU only
[[nodiscard]] int k_CountUnique(unsigned int *keys, int n);

inline void k_Unique(int *n_unique, unsigned int *keys, const int n) {
  *n_unique = k_CountUnique(keys, n);
}
