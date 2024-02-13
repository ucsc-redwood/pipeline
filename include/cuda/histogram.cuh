#pragma once

__global__ void k_GlobalHistogram(const unsigned int *sort,
                                  unsigned int *globalHistogram,
                                  int size,
                                  int logical_blocks);

__global__ void k_GlobalHistogram_Original(const unsigned int *sort,
                                           unsigned int *globalHistogram,
                                           int size);
