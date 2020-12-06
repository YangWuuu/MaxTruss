#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <algorithm>

#include "log.h"
#include "util.h"

__global__ void CoreKernel(const EdgeT *nodeIndex, NodeT *core, NodeT nodesNum) {
  auto from = blockDim.x * blockIdx.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;
  for (NodeT i = from; i < nodesNum; i += step) {
    core[i] = nodeIndex[i + 1] - nodeIndex[i];
  }
}

// 并行扫描度取值
__global__ void ScanKernel(NodeT nodesNum, const NodeT *core, NodeT level, NodeT *curr, NodeT *currTail) {
  auto from = blockDim.x * blockIdx.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;
  for (EdgeT i = from; i < nodesNum; i += step) {
    if (core[i] == level) {
      curr[atomicAdd(currTail, 1)] = i;
    }
  }
}

// 子任务循环迭代分解
__global__ void SubLevelKernel(const EdgeT *nodeIndex, const NodeT *adj, const NodeT *curr, NodeT currTail, NodeT *core,
                               NodeT level, NodeT *next, NodeT *nextTail) {
  for (NodeT i = blockIdx.x; i < currTail; i += gridDim.x) {
    NodeT u = curr[i];
    EdgeT uStart = nodeIndex[u];
    EdgeT uEnd = nodeIndex[u + 1];
    for (EdgeT j = uStart + threadIdx.x; j < uEnd; j += blockDim.x) {
      NodeT v = adj[j];
      NodeT degV = core[v];

      if (degV > level) {
        NodeT dv = atomicSub(&core[v], 1);
        if (dv == (level + 1)) {
          next[atomicAdd(nextTail, 1)] = v;
        }
      }
      __syncthreads();
    }
  }
}

__global__ void ProcessNextKernel(NodeT *core, NodeT level, const NodeT *next, NodeT nextTail) {
  auto from = blockDim.x * blockIdx.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;
  for (NodeT i = from; i < nextTail; i += step) {
    NodeT u = next[i];
    core[u] = level;
  }
}

// 求解k-core的主流程
NodeT KCore(const EdgeT *nodeIndex, const NodeT *adj, NodeT nodesNum, NodeT *&core) {
  CUDA_TRY(cudaMallocManaged((void **)&core, nodesNum * sizeof(NodeT)));

  CoreKernel<<<DIV_ROUND_UP(nodesNum, BLOCK_SIZE), BLOCK_SIZE>>>(nodeIndex, core, nodesNum);
  CUDA_TRY(cudaDeviceSynchronize());

  EdgeT *currTail;
  EdgeT *nextTail;
  CUDA_TRY(cudaMallocManaged((void **)&currTail, sizeof(EdgeT)));
  CUDA_TRY(cudaMallocManaged((void **)&nextTail, sizeof(EdgeT)));
  *currTail = 0;
  *nextTail = 0;

  EdgeT *curr;
  EdgeT *next;
  CUDA_TRY(cudaMallocManaged((void **)&curr, nodesNum * sizeof(EdgeT)));
  CUDA_TRY(cudaMallocManaged((void **)&next, nodesNum * sizeof(EdgeT)));

  NodeT todo = nodesNum;
  NodeT level = 0;

  while (todo > 0) {
    ScanKernel<<<DIV_ROUND_UP(nodesNum, BLOCK_SIZE), BLOCK_SIZE>>>(nodesNum, core, level, curr, currTail);
    CUDA_TRY(cudaDeviceSynchronize());
    log_debug("level: %u currTail: %u restNodes: %u", level, *currTail, todo);
    while (*currTail > 0) {
      todo = todo - *currTail;
      SubLevelKernel<<<*currTail, BLOCK_SIZE>>>(nodeIndex, adj, curr, *currTail, core, level, next, nextTail);
      CUDA_TRY(cudaDeviceSynchronize());

      if (*nextTail > 0) {
        ProcessNextKernel<<<DIV_ROUND_UP(*nextTail, BLOCK_SIZE), BLOCK_SIZE>>>(core, level, next, *nextTail);
        ProcessNextKernel<<<*nextTail, 1>>>(core, level, next, *nextTail);
        CUDA_TRY(cudaDeviceSynchronize());
      }

      std::swap(curr, next);
      *currTail = *nextTail;
      *nextTail = 0;

      log_debug("level: %u currTail: %u restNodes: %u", level, *currTail, todo);
    }
    level = level + 1;
  }

  CUDA_TRY(cudaFree(currTail));
  CUDA_TRY(cudaFree(nextTail));
  CUDA_TRY(cudaFree(curr));
  CUDA_TRY(cudaFree(next));

  thrust::device_ptr<NodeT> corePtr(core);
  return *thrust::max_element(corePtr, corePtr + nodesNum);
}
