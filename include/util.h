#pragma once

#include <cstdint>

#include "log.h"

#define FIRST(x) (uint32_t(x >> 32u))
#define SECOND(x) (uint32_t(x & 0xFFFFFFFF))
#define MAKE_EDGE(x, y) ((uint64_t)(uint64_t(x) << 32u | uint64_t(y)))

using NodeT = uint32_t;
using EdgeT = uint32_t;

// Size of cache line
const NodeT BUFFER_SIZE_BYTES = 2048;
const NodeT BUFFER_SIZE = BUFFER_SIZE_BYTES / sizeof(NodeT);

const uint64_t SEARCH_LINE_NUM = 1000u;

#define CUDA_TRY(call)                                                         \
  do {                                                                         \
    cudaError_t const status = (call);                                         \
    if (cudaSuccess != status) {                                               \
      log_error("%s %s %d\n", cudaGetErrorString(status), __FILE__, __LINE__); \
    }                                                                          \
  } while (0)

void *MyMalloc(uint64_t len);

void MyFree(void *&data, uint64_t len);
