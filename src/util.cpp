#include "util.h"

#include <fcntl.h>
#include <sys/mman.h>

// 在某些机器上表现比较好
void *myMalloc(uint64_t len) {
  int mmap_flags = MAP_ANONYMOUS | MAP_PRIVATE;
  void *data =
      (void *)mmap(nullptr, len, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  madvise(data, len, MADV_HUGEPAGE);
  return data;
}
