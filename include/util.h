#pragma once

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "log.h"

#define FIRST(x) (uint32_t(x >> 32u))
#define SECOND(x) (uint32_t(x & 0xFFFFFFFF))
#define MAKE_EDGE(x, y) ((uint64_t)(uint64_t(x) << 32u | uint64_t(y)))

using NodeT = uint32_t;
using EdgeT = uint32_t;

// Size of cache line
const EdgeT BUFFER_SIZE_BYTES = 2048;
const EdgeT BUFFER_SIZE = BUFFER_SIZE_BYTES / sizeof(NodeT);

const uint64_t FILE_SPLIT_NUM = 8u;
const uint64_t SEARCH_LINE_NUM = 1000u;
const int CACHE_LINE_ENTRY = 16;
const uint32_t SHRINK_SIZE = 7u;

inline std::string GetFileName(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s -f graph_name\n", argv[0]);
    exit(-1);
  }
  return std::string(argv[2]);
}


/*
 * InclusivePrefixSumOMP: General Case Inclusive Prefix Sum
 * histogram: is for cache-aware thread-local histogram purpose
 * output: should be different from the variables captured in function object
 * size: is the original size for the flagged prefix sum
 * f: requires it as the parameter, f(it) return the histogram value of that
 it
 */
template <typename H, typename T, typename F>
void InclusivePrefixSumOMP(std::vector<H> &histogram, T *output, size_t size,
                           F f) {
  int omp_num_threads = omp_get_num_threads();

#pragma omp single
  { histogram = std::vector<H>((omp_num_threads + 1) * CACHE_LINE_ENTRY, 0); }
  static thread_local int tid = omp_get_thread_num();
  // 1st Pass: Histogram.
  auto avg = size / omp_num_threads;
  auto it_beg = avg * tid;
  auto histogram_idx = (tid + 1) * CACHE_LINE_ENTRY;
  histogram[histogram_idx] = 0;
  auto it_end = tid == omp_num_threads - 1 ? size : avg * (tid + 1);
  size_t prev = 0u;
  for (auto it = it_beg; it < it_end; it++) {
    auto value = f(it);
    histogram[histogram_idx] += value;
    prev += value;
    output[it] = prev;
  }
#pragma omp barrier

  // 2nd Pass: single-prefix-sum & Add previous sum.
#pragma omp single
  {
    for (auto local_tid = 0; local_tid < omp_num_threads; local_tid++) {
      auto local_histogram_idx = (local_tid + 1) * CACHE_LINE_ENTRY;
      auto prev_histogram_idx = (local_tid)*CACHE_LINE_ENTRY;
      histogram[local_histogram_idx] += histogram[prev_histogram_idx];
    }
  }
  {
    auto prev_sum = histogram[tid * CACHE_LINE_ENTRY];
    for (auto it = it_beg; it < it_end; it++) {
      output[it] += prev_sum;
    }
#pragma omp barrier
  }
}

template <typename T>
uint32_t BranchFreeBinarySearch(const T *a, const uint32_t offset_beg,
                                const uint32_t offset_end, T x) {
  int32_t n = offset_end - offset_beg;
  using I = uint32_t;
  const T *base = a + offset_beg;
  while (n > 1) {
    I half = n / 2;
    __builtin_prefetch(base + half / 2, 0, 0);
    __builtin_prefetch(base + half + half / 2, 0, 0);
    base = (base[half] < x) ? base + half : base;
    n -= half;
  }
  return (*base < x) + base - a;
}

// Assuming (offset_beg != offset_end)
template <typename T>
uint32_t GallopingSearch(const T *array, const uint32_t offset_beg,
                         const uint32_t offset_end, T val) {
  if (array[offset_end - 1] < val) {
    return offset_end;
  }
  // galloping
  if (array[offset_beg] >= val) {
    return offset_beg;
  }
  if (array[offset_beg + 1] >= val) {
    return offset_beg + 1;
  }
  if (array[offset_beg + 2] >= val) {
    return offset_beg + 2;
  }

  auto jump_idx = 4u;
  while (true) {
    auto peek_idx = offset_beg + jump_idx;
    if (peek_idx >= offset_end) {
      return BranchFreeBinarySearch(array, (jump_idx >> 1u) + offset_beg + 1,
                                    offset_end, val);
    }
    if (array[peek_idx] < val) {
      jump_idx <<= 1u;
    } else {
      return array[peek_idx] == val
          ? peek_idx
          : BranchFreeBinarySearch(array,
                                   (jump_idx >> 1u) + offset_beg + 1,
                                   peek_idx + 1, val);
    }
  }
}

template <typename T>
uint32_t LinearSearch(const T *array, const uint32_t offset_beg,
                      const uint32_t offset_end, T val) {
  // linear search fallback
  for (auto offset = offset_beg; offset < offset_end; offset++) {
    if (array[offset] >= val) {
      return offset;
    }
  }
  return offset_end;
}
