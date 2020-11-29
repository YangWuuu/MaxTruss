#include <cstdlib>

#include "log.h"
#include "util.h"

void InitCuda() {
  cudaSetDevice(0);
  cudaFree(0);
}

__global__ void CUDAGetEdgeSup(EdgeT halfEdgesNum, const NodeT *halfEdgesFirst, const NodeT *halfEdgesSecond,
                               const EdgeT *halfNodeIndex, NodeT *edgesSup) {
  auto from = blockDim.x * blockIdx.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;
  for (EdgeT i = from; i < halfEdgesNum; i += step) {
    NodeT u = halfEdgesFirst[i];
    NodeT v = halfEdgesSecond[i];
    EdgeT uStart = halfNodeIndex[u];
    EdgeT uEnd = halfNodeIndex[u + 1];
    EdgeT vStart = halfNodeIndex[v];
    EdgeT vEnd = halfNodeIndex[v + 1];
    while (uStart < uEnd && vStart < vEnd) {
      if (halfEdgesSecond[uStart] < halfEdgesSecond[vStart]) {
        ++uStart;
      } else if (halfEdgesSecond[uStart] > halfEdgesSecond[vStart]) {
        ++vStart;
      } else {
        atomicAdd(edgesSup + i, 1);
        atomicAdd(edgesSup + uStart, 1);
        atomicAdd(edgesSup + vStart, 1);
        ++uStart;
        ++vStart;
      }
    }
  }
}

// 三角形计数获取支持边数量
void GetEdgeSup(EdgeT halfEdgesNum, const NodeT *halfEdgesFirst, const NodeT *halfEdgesSecond,
                const EdgeT *halfNodeIndex, NodeT nodesNum, NodeT *&edgesSup) {
  NodeT *cudaHalfEdgesFirst;
  NodeT *cudaHalfEdgesSecond;
  EdgeT *cudaHalfNodeIndex;
  NodeT *cudaEdgesSup;
  CUDA_TRY(cudaMalloc((void **)&cudaHalfEdgesFirst, halfEdgesNum * sizeof(NodeT)));
  CUDA_TRY(cudaMemcpy(cudaHalfEdgesFirst, halfEdgesFirst, halfEdgesNum * sizeof(NodeT), cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMalloc((void **)&cudaHalfEdgesSecond, halfEdgesNum * sizeof(NodeT)));
  CUDA_TRY(cudaMemcpy(cudaHalfEdgesSecond, halfEdgesSecond, halfEdgesNum * sizeof(NodeT), cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMalloc((void **)&cudaHalfNodeIndex, (nodesNum + 1) * sizeof(EdgeT)));
  CUDA_TRY(cudaMemcpy(cudaHalfNodeIndex, halfNodeIndex, (nodesNum + 1) * sizeof(EdgeT), cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMalloc((void **)&cudaEdgesSup, halfEdgesNum * sizeof(NodeT)));
  log_info("1");
  CUDAGetEdgeSup<<<(halfEdgesNum + 127) / 128, 128>>>(halfEdgesNum, cudaHalfEdgesFirst, cudaHalfEdgesSecond,
                                                      cudaHalfNodeIndex, cudaEdgesSup);
  CUDA_TRY(cudaDeviceSynchronize());
  edgesSup = (NodeT *)calloc(halfEdgesNum, sizeof(NodeT));
  CUDA_TRY(cudaMemcpy(edgesSup, cudaEdgesSup, halfEdgesNum * sizeof(NodeT), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaDeviceSynchronize());

  CUDA_TRY(cudaFree(cudaHalfEdgesFirst));
  CUDA_TRY(cudaFree(cudaHalfEdgesSecond));
  CUDA_TRY(cudaFree(cudaHalfNodeIndex));
  CUDA_TRY(cudaFree(cudaEdgesSup));
  CUDA_TRY(cudaDeviceSynchronize());
}
