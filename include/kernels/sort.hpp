#pragma once

void k_SortKeysInplace(unsigned int *keys, int n);

void k_SimpleRadixSort(unsigned int *keys, int n);
void k_SimpleRadixSort(unsigned int *keys, unsigned int *keys_alt, int n);

namespace gpu {
void k_CubRadixSort(const unsigned int *keys, unsigned int *keys_alt, int n);
}