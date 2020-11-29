#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>

#include "clock.h"
#include "log.h"

class ReadFile {
 public:
  explicit ReadFile(const std::string &filePath) : filePath_(filePath), readFileClock_("ReadFile") {}

  ~ReadFile() { Release(); }

  uint64_t ConstructEdges(uint64_t *&edges);

 private:
  // 读取文件
  void GetData() {
    struct stat statBuf {};
    if ((fd_ = open(filePath_.c_str(), O_RDONLY)) < 0) {
      fprintf(stderr, "open file error\n");
      exit(-1);
    }
    if ((fstat(fd_, &statBuf)) < 0) {
      fprintf(stderr, "get file length error\n");
      exit(-1);
    }
    len_ = statBuf.st_size;
    if ((byte_ = (char *)mmap(nullptr, len_, PROT_READ, MAP_SHARED, fd_, 0)) == (void *)-1) {
      fprintf(stderr, "mmap file error\n");
      exit(-1);
    }
  }

  // 释放文件
  void Release() {
    if (fd_ != -1) {
      close(fd_);
      munmap(byte_, len_);
    }
    fd_ = -1;
    byte_ = nullptr;
  }

  uint64_t GetLineNum();
  uint64_t GetEdges(uint64_t *edges);

 private:
  const std::string &filePath_;
  Clock readFileClock_;

  int fd_{-1};
  char *byte_{nullptr};
  uint64_t len_{0};
};

uint64_t GetLineNum(const char *byte, uint64_t len);
uint64_t GetEdges(uint64_t *edges, const char *byte, uint64_t len);
