#pragma once

// __global__ void k_DigitBinning(unsigned int* globalHistogram,
//                                unsigned int* sort,
//                                unsigned int* alt,
//                                volatile unsigned int* passHistogram,
//                                unsigned int* index,
//                                int size,
//                                unsigned int radixShift,
//                                int logical_blocks);

__global__ void k_DigitBinning_Original(const unsigned int* globalHistogram,
                                        const unsigned int* sort,
                                        unsigned int* alt,
                                        volatile unsigned int* passHistogram,
                                        unsigned int* index,
                                        int size,
                                        unsigned int radixShift);
