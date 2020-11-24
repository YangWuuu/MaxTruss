#pragma once

#include <cstdint>

#define FIRST(x) (uint32_t(x >> 32u))
#define SECOND(x) (uint32_t(x & 0xFFFFFFFF))
#define MAKE_EDGE(x, y) ((uint64_t)(uint64_t(x) << 32u | uint64_t(y)))

using NodeT = uint32_t;
using EdgeT = uint32_t;

// Size of cache line
const NodeT BUFFER_SIZE_BYTES = 2048;
const NodeT BUFFER_SIZE = BUFFER_SIZE_BYTES / sizeof(NodeT);

const uint64_t SEARCH_LINE_NUM = 1000u;

void *myMalloc(uint64_t len);
