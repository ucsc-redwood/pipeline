#include <omp.h>

#include <algorithm>

constexpr int BASE_BITS = 8;
constexpr int BASE = (1 << BASE_BITS);
constexpr int MASK = (BASE - 1);

inline auto DIGITS(unsigned int v, int shift) -> int {
  return (v >> shift) & MASK;
}

inline void omp_lsd_radix_sort(int n, unsigned int* data) {
  auto* buffer =
      static_cast<unsigned int*>(std::malloc(n * sizeof(unsigned int)));
  constexpr auto total_digits = sizeof(unsigned int) * 8;

  for (auto shift = 0; shift < total_digits; shift += BASE_BITS) {
    int bucket[BASE] = {0};
    int local_bucket[BASE] = {0};  // size needed in each bucket/thread

#pragma omp parallel firstprivate(local_bucket)
    {
#pragma omp for schedule(static) nowait
      for (auto i = 0; i < n; i++) {
        local_bucket[DIGITS(data[i], shift)]++;
      }
#pragma omp critical
      for (auto i = 0; i < BASE; i++) {
        bucket[i] += local_bucket[i];
      }
#pragma omp barrier
#pragma omp single
      for (auto i = 1; i < BASE; i++) {
        bucket[i] += bucket[i - 1];
      }
      auto nthreads = omp_get_num_threads();
      auto tid = omp_get_thread_num();
      for (auto cur_t = nthreads - 1; cur_t >= 0; cur_t--) {
        if (cur_t == tid) {
          for (auto i = 0; i < BASE; i++) {
            bucket[i] -= local_bucket[i];
            local_bucket[i] = bucket[i];
          }
        } else {
#pragma omp barrier
        }
      }
#pragma omp for schedule(static)
      for (auto i = 0; i < n; i++) {
        buffer[local_bucket[DIGITS(data[i], shift)]++] = data[i];
      }
    }
    // now move data
    std::swap(data, buffer);
  }
  std::free(buffer);
}

inline void omp_lsd_radix_sort(int n,
                               unsigned int* data,
                               unsigned int* data_alt) {
  constexpr auto total_digits = sizeof(unsigned int) * 8;

  for (auto shift = 0; shift < total_digits; shift += BASE_BITS) {
    int bucket[BASE] = {0};
    int local_bucket[BASE] = {0};  // size needed in each bucket/thread

#pragma omp parallel firstprivate(local_bucket)
    {
#pragma omp for schedule(static) nowait
      for (auto i = 0; i < n; i++) {
        local_bucket[DIGITS(data[i], shift)]++;
      }
#pragma omp critical
      for (auto i = 0; i < BASE; i++) {
        bucket[i] += local_bucket[i];
      }
#pragma omp barrier
#pragma omp single
      for (auto i = 1; i < BASE; i++) {
        bucket[i] += bucket[i - 1];
      }
      auto nthreads = omp_get_num_threads();
      auto tid = omp_get_thread_num();
      for (auto cur_t = nthreads - 1; cur_t >= 0; cur_t--) {
        if (cur_t == tid) {
          for (auto i = 0; i < BASE; i++) {
            bucket[i] -= local_bucket[i];
            local_bucket[i] = bucket[i];
          }
        } else {
#pragma omp barrier
        }
      }
#pragma omp for schedule(static)
      for (auto i = 0; i < n; i++) {
        data_alt[local_bucket[DIGITS(data[i], shift)]++] = data[i];
      }
    }
    // now move data
    std::swap(data, data_alt);
  }
}

void k_SortKeysInplace(unsigned int* keys, const int n) {
  std::sort(keys, keys + n);
}

void k_SimpleRadixSort(unsigned int* keys, const int n) {
  omp_lsd_radix_sort(n, keys);
}

void k_SimpleRadixSort(unsigned int* keys,
                       unsigned int* keys_alt,
                       const int n) {
  omp_lsd_radix_sort(n, keys, keys_alt);
}
