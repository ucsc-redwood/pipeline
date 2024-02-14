#pragma once

namespace gpu {

// Don't use them directly, use the wrapper functions below

__global__ void k_GlobalHistogram(const unsigned int* sort,
                                  unsigned int* globalHistogram,
                                  int size,
                                  int logical_blocks);

__global__ void k_DigitBinning(const unsigned int* globalHistogram,
                               const unsigned int* sort,
                               unsigned int* alt,
                               volatile unsigned int* passHistogram,
                               unsigned int* index,
                               int size,
                               unsigned int radixShift);

void DispatchRadixSort();

}  // namespace gpu