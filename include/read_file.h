#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "clock.h"
#include "log.h"
#include "util.h"

class ReadFile {
 public:
  ReadFile(const std::string &filePath, Clock &readFileClock)
      : filePath_(filePath), readFileClock_(readFileClock) {}

  ~ReadFile() { Release(); }

  uint64_t ConstructEdges(uint64_t *&edges) {
    log_info(readFileClock_.Count("Start"));
    GetData();
    log_info(readFileClock_.Count("GetData"));
    uint64_t lineNum = GetLineNum();
    log_info(readFileClock_.Count("lineNum: %lu", lineNum));
    edges = (uint64_t *)malloc((lineNum + 1) * sizeof(uint64_t));
    log_info(readFileClock_.Count("edges malloc"));
    uint64_t edgesNum = GetEdges(edges);
    log_info(readFileClock_.Count("edgesNum: %lu", edgesNum));
    bool isEdgesSorted = std::is_sorted(edges, edges + edgesNum);
    log_info(readFileClock_.Count("isEdgesSorted: %s",
                                  isEdgesSorted ? "sorted" : "no sorted"));
    if (!isEdgesSorted) {
      uint64_t newEdgesNum = edgesNum * 2;
      auto newEdges = (uint64_t *)malloc(newEdgesNum * sizeof(uint64_t));
      memcpy(newEdges, edges, edgesNum * sizeof(uint64_t));
      for (uint64_t i = 0; i < edgesNum; i++) {
        newEdges[edgesNum + i] = MAKEEDGE(SECOND(edges[i]), FIRST(edges[i]));
      }
      edges = newEdges;
      edgesNum = newEdgesNum;
      log_info(readFileClock_.Count("newEdgesNum: %lu", newEdgesNum));
      std::sort(edges, edges + edgesNum);
      log_info(readFileClock_.Count("edges sort"));
    }
    edgesNum = std::unique(edges, edges + edgesNum) - edges;
    log_info(readFileClock_.Count("unique edgesNum: %lu", edgesNum));
    edgesNum = std::remove_if(edges, edges + edgesNum,
                              [](const uint64_t &edge) {
                                return FIRST(edge) == SECOND(edge);
                              }) -
               edges;
    log_info(readFileClock_.Count("remove selfLoop edgesNum: %lu", edgesNum));
    return edgesNum;
  }

 private:
  void Release() {
    if (fd_ != -1) {
      close(fd_);
      munmap(byte_, len_);
    }
    fd_ = -1;
    byte_ = nullptr;
  }

  void GetData() {
    struct stat statBuf {};
    if ((fd_ = open(filePath_.c_str(), O_RDONLY)) < 0) {
      log_error("open file error");
      exit(-1);
    }
    if ((fstat(fd_, &statBuf)) < 0) {
      log_error("get file length error");
      exit(-1);
    }
    len_ = statBuf.st_size;
    if ((byte_ = (char *)mmap(nullptr, len_, PROT_READ, MAP_SHARED, fd_, 0)) ==
        (void *)-1) {
      log_error("mmap file error");
      exit(-1);
    }
  }

  uint64_t GetLineNum() {
    if (!byte_) {
      log_error("byte_ is nullptr");
      exit(-1);
    }
    uint64_t pos = 0;
    uint64_t lineNum = 0;
    while (pos < len_) {
      if (*(byte_ + pos) == '\n') {
        ++lineNum;
        pos += 4;  // one line has at least 6 char
      }
      ++pos;
    }
    return lineNum;
  }

  uint64_t GetEdges(uint64_t *edges) {
    uint64_t pos = 0;
    uint64_t edgesNum = 0;
    while (pos < len_) {
      uint32_t first = 0;
      while (*(byte_ + pos) != '\t') {
        first = 10u * first + (*(byte_ + pos) - '0');
        ++pos;
      }
      ++pos;
      uint32_t second = 0;
      while (*(byte_ + pos) != '\t') {
        second = 10u * second + (*(byte_ + pos) - '0');
        ++pos;
      }
      while (*(byte_ + pos) != '\n') {
        ++pos;
      }
      ++pos;
      edges[edgesNum] = MAKEEDGE(second, first);
      ++edgesNum;
    }
    return edgesNum;
  }

 private:
  const std::string &filePath_;
  Clock &readFileClock_;

  int fd_{-1};
  char *byte_{nullptr};
  uint64_t len_{0};
};
