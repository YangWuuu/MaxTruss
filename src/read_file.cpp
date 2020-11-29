#include "read_file.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <vector>

#include "util.h"

#pragma ide diagnostic ignored "openmp-use-default-none"

uint64_t ReadFile::ConstructEdges(uint64_t *&edges) {
  log_info(readFileClock_.Start());

  GetData();
  log_info(readFileClock_.Count("GetData"));

#ifdef SERIAL
  uint64_t lineNum = ::GetLineNum(byte_, len_);
  log_info(readFileClock_.Count("lineNum: %lu", lineNum));
#else
  uint64_t lineNum = GetLineNum();
  log_info(readFileClock_.Count("approximateLineNum: %lu", lineNum));
#endif

  edges = (uint64_t *)MyMalloc((lineNum + 1) * sizeof(uint64_t));
  log_info(readFileClock_.Count("edges: %.2fMB", (double)lineNum * sizeof(uint64_t) / 1024 / 1024));

  uint64_t edgesNum = GetEdges(edges);
  log_info(readFileClock_.Count("edgesNum: %lu %.2fMB", edgesNum, (double)edgesNum * sizeof(uint64_t) / 1024 / 1024));

  return edgesNum;
}

uint64_t ReadFile::GetLineNum() {
  if (!byte_) {
    fprintf(stderr, "byte_ is nullptr");
    exit(-1);
  }

  uint64_t len = std::min(len_, SEARCH_LINE_NUM * (10 * 2 + 4));
  uint64_t lineNum = ::GetLineNum(byte_, len);
  uint64_t approximateLineNum = std::ceil(1.2 * len_ * lineNum / len);

  return approximateLineNum;
}

uint64_t ReadFile::GetEdges(uint64_t *edges) {
  uint64_t edgesNum = 0;

#ifdef SERIAL
  edgesNum = ::GetEdges(edges, byte_, len_);
#else
  uint64_t fileSplitNum = 8u;
  if (len_ < 1000u) {
    fileSplitNum = 1u;
  }
  std::vector<std::thread> threads(fileSplitNum);
  for (uint64_t i = 0; i < fileSplitNum; i++) {
    uint64_t start = len_ * i / fileSplitNum;
    uint64_t end = len_ * (i + 1) / fileSplitNum;
    if (i != 0) {
      while (*(byte_ + start) != '\n') {
        ++start;
      }
      ++start;
    }
    if (i + 1 != fileSplitNum) {
      while (*(byte_ + end) != '\n') {
        ++end;
      }
      ++end;
    }
    threads[i] = std::thread([=]() { ::GetEdges(edges + edgesNum, byte_ + start, end - start); });
    edgesNum += ::GetLineNum(byte_ + start, end - start);
  }
  log_info(readFileClock_.Count("before join fileSplitNum: %u", fileSplitNum));
  for (auto &thread : threads) {
    thread.join();
  }
#endif

  return edgesNum;
}

uint64_t GetLineNum(const char *byte, uint64_t len) {
  if (!byte) {
    log_error("byte_ is nullptr");
    exit(-1);
  }
  while (*(byte + len - 1) == '\n' || *(byte + len - 1) == '\r') {
    --len;
  }
  uint64_t lineNum = 0;

#pragma omp parallel for reduction(+ : lineNum)
  for (uint64_t pos = 0; pos < len; pos++) {
    if (*(byte + pos) == '\n') {
      ++lineNum;
    }
  }
  if (*(byte + len - 1) != '\n') {
    ++lineNum;
  }
  return lineNum;
}

uint64_t GetEdges(uint64_t *edges, const char *byte, uint64_t len) {
  while (*(byte + len - 1) == '\n' || *(byte + len - 1) == '\r') {
    --len;
  }

  uint64_t pos = 0;
  uint64_t edgesNum = 0;
  while (pos < len) {
    uint32_t first = 0;
    while (*(byte + pos) >= '0' && *(byte + pos) <= '9') {
      first = 10u * first + (*(byte + pos) - '0');
      ++pos;
    }
    ++pos;
    uint32_t second = 0;
    while (*(byte + pos) >= '0' && *(byte + pos) <= '9') {
      second = 10u * second + (*(byte + pos) - '0');
      ++pos;
    }
    while (pos < len && *(byte + pos) != '\n') {
      ++pos;
    }
    ++pos;
    edges[edgesNum] = MAKE_EDGE(second, first);
    ++edgesNum;
  }
  return edgesNum;
}
