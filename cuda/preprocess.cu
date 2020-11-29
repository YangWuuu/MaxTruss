#include <algorithm>
#include <cstdlib>

#include "util.h"

#pragma ide diagnostic ignored "openmp-use-default-none"

// 计算节点的度
void CalDeg(const uint64_t *edges, EdgeT edgesNum, NodeT nodesNum, NodeT *&deg) {
  deg = (NodeT *)calloc(nodesNum, sizeof(NodeT));
#ifdef SERIAL
  for (EdgeT i = 0; i < edgesNum; i++) {
    ++deg[FIRST(edges[i])];
  }
#else
#pragma omp parallel for
  for (EdgeT i = 0; i < edgesNum; i++) {
    __sync_fetch_and_add(&deg[FIRST(edges[i])], 1);
  }
#endif
}

// 边的解压缩
void Unzip(const uint64_t *edges, EdgeT edgesNum, NodeT *&edgesFirst, NodeT *&edgesSecond) {
  edgesFirst = (NodeT *)MyMalloc(edgesNum * sizeof(NodeT));
  edgesSecond = (NodeT *)MyMalloc(edgesNum * sizeof(NodeT));

#pragma omp parallel for
  for (EdgeT i = 0; i < edgesNum; i++) {
    edgesFirst[i] = FIRST(edges[i]);
    edgesSecond[i] = SECOND(edges[i]);
  }
}

// 转换CSR格式
void NodeIndex(const NodeT *deg, NodeT nodesNum, EdgeT *&nodeIndex) {
  nodeIndex = (EdgeT *)calloc((nodesNum + 1), sizeof(EdgeT));
  // 这里并行不一定比串行快，涉及到伪共享问题
  for (NodeT i = 0; i < nodesNum; i++) {
    nodeIndex[i + 1] = nodeIndex[i] + deg[i];
  }
}

__inline__ __device__ EdgeT lowerBound(const NodeT *arr, NodeT start, NodeT end, NodeT value) {
  while (start <= end) {
    int middle = start + (end - start) / 2;
    if (value > arr[middle]) {
      start = middle + 1;
    } else if (value < arr[middle]) {
      end = middle - 1;
    } else
      return middle;
  }
  return 0;
}

__global__ void CUDAGetEdgesId(const uint64_t *edges, EdgeT edgesNum, EdgeT *edgesId, const EdgeT *halfNodeIndex,
                               const NodeT *halfEdgesSecond) {
  auto from = blockDim.x * blockIdx.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;
  for (EdgeT i = from; i < edgesNum; i += step) {
    NodeT u = FIRST(edges[i]);
    NodeT v = SECOND(edges[i]);
    if (u > v) {
      NodeT tmp = u;
      u = v;
      v = tmp;
    }
    edgesId[i] = lowerBound(halfEdgesSecond, halfNodeIndex[u], halfNodeIndex[u + 1], v);
  }
}

// 边编号
void GetEdgesId(const uint64_t *edges, EdgeT edgesNum, EdgeT *&edgesId, const EdgeT *halfNodeIndex, NodeT nodesNum,
                const NodeT *halfEdgesSecond) {
  uint64_t *cudaEdges;
  EdgeT *cudaHalfNodeIndex;
  NodeT *cudaHalfEdgesSecond;
  EdgeT *cudaEdgesId;
  CUDA_TRY(cudaMalloc((void **)&cudaEdges, edgesNum * sizeof(uint64_t)));
  CUDA_TRY(cudaMemcpy(cudaEdges, edges, edgesNum * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMalloc((void **)&cudaHalfNodeIndex, edgesNum * sizeof(EdgeT)));
  CUDA_TRY(cudaMemcpy(cudaHalfNodeIndex, edges, edgesNum * sizeof(EdgeT), cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMalloc((void **)&cudaHalfEdgesSecond, edgesNum * sizeof(NodeT)));
  CUDA_TRY(cudaMemcpy(cudaHalfEdgesSecond, halfEdgesSecond, edgesNum * sizeof(NodeT), cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMalloc((void **)&cudaEdgesId, edgesNum * sizeof(EdgeT)));
  CUDA_TRY(cudaDeviceSynchronize());
  log_info("1");
  CUDAGetEdgesId<<<(edgesNum + 127) / 128, 128>>>(cudaEdges, edgesNum, cudaEdgesId, cudaHalfNodeIndex,
                                                  cudaHalfEdgesSecond);
  CUDA_TRY(cudaDeviceSynchronize());
  log_info("2");
  edgesId = (EdgeT *)MyMalloc(edgesNum * sizeof(EdgeT));
  CUDA_TRY(cudaMemcpy(edgesId, cudaEdgesId, edgesNum * sizeof(EdgeT), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaDeviceSynchronize());
  CUDA_TRY(cudaFree(cudaEdges));
  CUDA_TRY(cudaFree(cudaHalfNodeIndex));
  CUDA_TRY(cudaFree(cudaHalfEdgesSecond));
  CUDA_TRY(cudaFree(cudaEdgesId));
  CUDA_TRY(cudaDeviceSynchronize());
}
