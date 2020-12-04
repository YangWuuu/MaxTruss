#include <cstdlib>

#include "log.h"
#include "util.h"

__global__ void CUDAGetEdgeSup(EdgeT halfEdgesNum, const NodeT *halfEdgesFirst, const NodeT *cudaAdj,
                               const EdgeT *halfNodeIndex, NodeT *edgesSup) {
  auto from = blockDim.x * blockIdx.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;
  for (EdgeT i = from; i < halfEdgesNum; i += step) {
    NodeT u = halfEdgesFirst[i];
    NodeT v = cudaAdj[i];
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

__global__ void UnzipEdgesKernel1(const uint64_t *edges, EdgeT edgesNum, NodeT *edgesFirst, NodeT *adj) {
  auto from = blockDim.x * blockIdx.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;
  for (EdgeT i = from; i < edgesNum; i += step) {
    uint64_t tmp = edges[i];
    edgesFirst[i] = FIRST(tmp);
    adj[i] = SECOND(tmp);
  }
}

// 三角形计数获取支持边数量
void GetEdgeSup(EdgeT halfEdgesNum, const uint64_t *halfEdges, const EdgeT *halfNodeIndex, NodeT nodesNum,
                NodeT *&edgesSup) {
  uint64_t *cudaHalfEdges;
  NodeT *cudaHalfEdgesFirst;
  NodeT *cudaAdj;
  EdgeT *cudaHalfNodeIndex;
  NodeT *cudaEdgesSup;
  CUDA_TRY(cudaMalloc((void **)&cudaHalfEdges, halfEdgesNum * sizeof(uint64_t)));
  CUDA_TRY(cudaMemcpy(cudaHalfEdges, halfEdges, halfEdgesNum * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMalloc((void **)&cudaHalfEdgesFirst, halfEdgesNum * sizeof(NodeT)));
  CUDA_TRY(cudaMalloc((void **)&cudaAdj, halfEdgesNum * sizeof(NodeT)));

  UnzipEdgesKernel1<<<(halfEdgesNum + 127) / 128, 128>>>(cudaHalfEdges, halfEdgesNum, cudaHalfEdgesFirst, cudaAdj);

  CUDA_TRY(cudaMalloc((void **)&cudaHalfNodeIndex, (nodesNum + 1) * sizeof(EdgeT)));
  CUDA_TRY(cudaMemcpy(cudaHalfNodeIndex, halfNodeIndex, (nodesNum + 1) * sizeof(EdgeT), cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMalloc((void **)&cudaEdgesSup, halfEdgesNum * sizeof(NodeT)));
  log_info("1");
  CUDAGetEdgeSup<<<(halfEdgesNum + 127) / 128, 128>>>(halfEdgesNum, cudaHalfEdgesFirst, cudaAdj, cudaHalfNodeIndex,
                                                      cudaEdgesSup);
  CUDA_TRY(cudaDeviceSynchronize());
  edgesSup = (NodeT *)calloc(halfEdgesNum, sizeof(NodeT));
  CUDA_TRY(cudaMemcpy(edgesSup, cudaEdgesSup, halfEdgesNum * sizeof(NodeT), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaDeviceSynchronize());

  CUDA_TRY(cudaFree(cudaHalfEdges));
  CUDA_TRY(cudaFree(cudaHalfEdgesFirst));
  CUDA_TRY(cudaFree(cudaAdj));
  CUDA_TRY(cudaFree(cudaHalfNodeIndex));
  CUDA_TRY(cudaFree(cudaEdgesSup));
  CUDA_TRY(cudaDeviceSynchronize());
}
