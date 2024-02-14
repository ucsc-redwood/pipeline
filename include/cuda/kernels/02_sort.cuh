#pragma once

#include <cstdint>

namespace gpu {

// Don't use them directly, use the wrapper functions below

__global__ void k_GlobalHistogram(uint32_t *sort,
                                  uint32_t *globalHistogram,
                                  uint32_t size);

__global__ void k_DigitBinning(uint32_t *globalHistogram,
                               uint32_t *sort,
                               uint32_t *alt,
                               volatile uint32_t *passHistogram,
                               uint32_t *index,
                               uint32_t size,
                               uint32_t radixShift);

__global__ void k_Print(unsigned int *toPrint, int size);

}  // namespace gpu