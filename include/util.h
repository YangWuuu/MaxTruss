#pragma once

#ifdef CUDA
#include <cuda_runtime.h>

#endif

#include <cstdint>
#include <limits>

#include "log.h"

#define FIRST(x) (uint32_t(x >> 32u))
#define SECOND(x) (uint32_t(x & 0xFFFFFFFF))
#define MAKE_EDGE(x, y) ((uint64_t)(uint64_t(x) << 32u | uint64_t(y)))
#define DIV_ROUND_UP(n, d) (((n) + (d)-1) / (d))

using NodeT = uint32_t;
using EdgeT = uint32_t;

// Size of cache line
const NodeT BUFFER_SIZE_BYTES = 2048;
const NodeT BUFFER_SIZE = BUFFER_SIZE_BYTES / sizeof(NodeT);

const uint64_t SEARCH_LINE_NUM = 1000u;

const uint32_t WARP_BITS = 5u;
const uint32_t WARP_SIZE = 1u << WARP_BITS;
const uint32_t WARP_MASK = WARP_SIZE - 1u;
const uint32_t BLOCK_SIZE = 128u;
const uint32_t WARPS_PER_BLOCK = (BLOCK_SIZE / WARP_SIZE);
const uint32_t GRID_SIZE = 1024u;
const EdgeT INVALID_NUM = std::numeric_limits<EdgeT>::max();

#define CUDA_TRY(call)                                                         \
  do {                                                                         \
    cudaError_t const status = (call);                                         \
    if (cudaSuccess != status) {                                               \
      log_error("%s %s %d\n", cudaGetErrorString(status), __FILE__, __LINE__); \
    }                                                                          \
  } while (0)

void *MyMalloc(uint64_t len);

void MyFree(void *&data, uint64_t len);
