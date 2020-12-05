#include <cstdlib>

#include "log.h"
#include "util.h"

__global__ void CUDAGetEdgeSup(EdgeT halfEdgesNum, NodeT halfNodesNum, const NodeT *halfEdgesFirst,
                               const NodeT *cudaAdj, const EdgeT *halfNodeIndex, NodeT *edgesSup) {
  auto from = blockDim.x * blockIdx.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;
  for (EdgeT i = from; i < halfEdgesNum; i += step) {
    NodeT u = halfEdgesFirst[i];
    NodeT v = cudaAdj[i];
    if (v >= halfNodesNum) {
      continue;
    }
    EdgeT uStart = halfNodeIndex[u];
    EdgeT uEnd = halfNodeIndex[u + 1];
    EdgeT vStart = halfNodeIndex[v];
    EdgeT vEnd = halfNodeIndex[v + 1];
    while (uStart < uEnd && vStart < vEnd) {
      if (cudaAdj[uStart] < cudaAdj[vStart]) {
        ++uStart;
      } else if (cudaAdj[uStart] > cudaAdj[vStart]) {
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

__global__ void ConstructEdgesFirstKernel(const EdgeT *nodeIndex, NodeT nodesNum, NodeT *edgesFirst) {
  auto from = blockDim.x * blockIdx.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;
  for (NodeT i = from; i < nodesNum; i += step) {
    for (EdgeT j = nodeIndex[i]; j < nodeIndex[i + 1]; ++j) {
      edgesFirst[j] = i;
    }
  }
}

// 三角形计数获取支持边数量
void GetEdgeSup(const EdgeT *halfNodeIndex, const NodeT *halfAdj, NodeT halfNodesNum, NodeT *&edgesSup) {
  EdgeT halfEdgesNum = halfNodeIndex[halfNodesNum];
  NodeT *halfEdgesFirst;

  CUDA_TRY(cudaMallocManaged((void **)&halfEdgesFirst, halfEdgesNum * sizeof(EdgeT)));

  ConstructEdgesFirstKernel<<<DIV_ROUND_UP(halfNodesNum, BLOCK_SIZE), BLOCK_SIZE>>>(halfNodeIndex, halfNodesNum,
                                                                                    halfEdgesFirst);
  CUDA_TRY(cudaDeviceSynchronize());

  CUDA_TRY(cudaMallocManaged((void **)&edgesSup, halfEdgesNum * sizeof(NodeT)));
  CUDAGetEdgeSup<<<DIV_ROUND_UP(halfEdgesNum, BLOCK_SIZE), BLOCK_SIZE>>>(halfEdgesNum, halfNodesNum, halfEdgesFirst,
                                                                         halfAdj, halfNodeIndex, edgesSup);
  CUDA_TRY(cudaDeviceSynchronize());

  CUDA_TRY(cudaFree(halfEdgesFirst));
}
