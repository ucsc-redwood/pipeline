#pragma once

// These are CPU only parrallel sorting function, on GPU we use 'one-sweep'

void k_SortKeysInplace(unsigned int *keys, int n);
void k_SimpleRadixSort(unsigned int *keys, int n);
void k_SimpleRadixSort(unsigned int *keys, unsigned int *keys_alt, int n);
