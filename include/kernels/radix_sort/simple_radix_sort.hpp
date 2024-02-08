#pragma once

// https://gist.github.com/wanghc78/2c2b403299cab172e74c62f4397a6997#file-openmp-radixsort-c-L54

#include <omp.h>

#define BASE_BITS 8
#define BASE (1 << BASE_BITS)
#define MASK (BASE - 1)
#define DIGITS(v, shift) (((v) >> shift) & MASK)

inline void omp_lsd_radix_sort(int n, unsigned int* data) {
  unsigned int* buffer = (unsigned int*)malloc(n * sizeof(unsigned int));
  int total_digits = sizeof(unsigned int) * 8;

  // Each thread use local_bucket to move data
  int i;
  for (int shift = 0; shift < total_digits; shift += BASE_BITS) {
    int bucket[BASE] = {0};

    int local_bucket[BASE] = {0};  // size needed in each bucket/thread
// 1st pass, scan whole and check the count
#pragma omp parallel firstprivate(local_bucket)
    {
#pragma omp for schedule(static) nowait
      for (i = 0; i < n; i++) {
        local_bucket[DIGITS(data[i], shift)]++;
      }
#pragma omp critical
      for (i = 0; i < BASE; i++) {
        bucket[i] += local_bucket[i];
      }
#pragma omp barrier
#pragma omp single
      for (i = 1; i < BASE; i++) {
        bucket[i] += bucket[i - 1];
      }
      int nthreads = omp_get_num_threads();
      int tid = omp_get_thread_num();
      for (int cur_t = nthreads - 1; cur_t >= 0; cur_t--) {
        if (cur_t == tid) {
          for (i = 0; i < BASE; i++) {
            bucket[i] -= local_bucket[i];
            local_bucket[i] = bucket[i];
          }
        } else {  // just do barrier
#pragma omp barrier
        }
      }
#pragma omp for schedule(static)
      for (i = 0; i < n; i++) {  // note here the end condition
        buffer[local_bucket[DIGITS(data[i], shift)]++] = data[i];
      }
    }
    // now move data
    unsigned int* tmp = data;
    data = buffer;
    buffer = tmp;
  }
  free(buffer);
}


inline void omp_lsd_radix_sort(int n, unsigned int* data, unsigned int* data_alt) {
  // unsigned int* buffer = (unsigned int*)malloc(n * sizeof(unsigned int));
  int total_digits = sizeof(unsigned int) * 8;

  // Each thread use local_bucket to move data
  int i;
  for (int shift = 0; shift < total_digits; shift += BASE_BITS) {
    int bucket[BASE] = {0};

    int local_bucket[BASE] = {0};  // size needed in each bucket/thread
// 1st pass, scan whole and check the count
#pragma omp parallel firstprivate(local_bucket)
    {
#pragma omp for schedule(static) nowait
      for (i = 0; i < n; i++) {
        local_bucket[DIGITS(data[i], shift)]++;
      }
#pragma omp critical
      for (i = 0; i < BASE; i++) {
        bucket[i] += local_bucket[i];
      }
#pragma omp barrier
#pragma omp single
      for (i = 1; i < BASE; i++) {
        bucket[i] += bucket[i - 1];
      }
      int nthreads = omp_get_num_threads();
      int tid = omp_get_thread_num();
      for (int cur_t = nthreads - 1; cur_t >= 0; cur_t--) {
        if (cur_t == tid) {
          for (i = 0; i < BASE; i++) {
            bucket[i] -= local_bucket[i];
            local_bucket[i] = bucket[i];
          }
        } else {  // just do barrier
#pragma omp barrier
        }
      }
#pragma omp for schedule(static)
      for (i = 0; i < n; i++) {  // note here the end condition
        data_alt[local_bucket[DIGITS(data[i], shift)]++] = data[i];
      }
    }
    // now move data
    unsigned int* tmp = data;
    data = data_alt;
    data_alt = tmp;
  }
  // free(buffer);
}