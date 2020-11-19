#include "log.h"

__global__ void add(int a, int b, int *c) { *c = a + b; }

void init_gpu() {
  cudaDeviceProp deviceProp{};
  cudaGetDeviceProperties(&deviceProp, 0);
  uint64_t gpu_mem = deviceProp.totalGlobalMem;
  log_info("Mem: %uMB", gpu_mem / 1024 / 1024);
}

int main() {
  int c;
  int *dev_c;
  cudaMalloc((void **)&dev_c, sizeof(int));
  add<<<1, 1>>>(2, 7, dev_c);
  cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
  log_info("2 + 7 = %d", c);
  cudaFree(dev_c);
  init_gpu();
  return 0;
}
