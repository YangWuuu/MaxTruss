#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>

#include <algorithm>
#include <cstdlib>

#include "util.h"

#pragma ide diagnostic ignored "openmp-use-default-none"

struct MoreThanCore : public thrust::unary_function<uint64_t, bool> {
  thrust::device_ptr<NodeT> rawCorePtr_;
  NodeT startK_;

  MoreThanCore(thrust::device_ptr<NodeT> rawCorePtr, NodeT startK) : rawCorePtr_(rawCorePtr), startK_(startK) {}
  __host__ __device__ bool operator()(uint64_t edge) {
    return rawCorePtr_[FIRST(edge)] >= (startK_ - 2) && rawCorePtr_[SECOND(edge)] >= (startK_ - 2);
  }
};

// 图的裁剪
EdgeT ConstructNewGraph(const uint64_t *rawEdges, uint64_t *&edges, const NodeT *cudaRawCore, EdgeT rawEdgesNum,
                        NodeT startK) {
  // TODO rawEdges 可能需要分块
  uint64_t *cudaRawEdges;
  uint64_t *cudaEdges;
  CUDA_TRY(cudaMalloc((void **)&cudaRawEdges, rawEdgesNum * sizeof(uint64_t)));
  CUDA_TRY(cudaMemcpy(cudaRawEdges, rawEdges, rawEdgesNum * sizeof(uint64_t), cudaMemcpyHostToDevice));

  thrust::device_ptr<uint64_t> cudaRawEdgesPtr(cudaRawEdges);

  thrust::device_ptr<NodeT> rawCorePtr(const_cast<NodeT *>(cudaRawCore));
  EdgeT edgesNum = thrust::count_if(cudaRawEdgesPtr, cudaRawEdgesPtr + rawEdgesNum, MoreThanCore(rawCorePtr, startK));

  CUDA_TRY(cudaMalloc((void **)&cudaEdges, edgesNum * sizeof(uint64_t)));
  thrust::device_ptr<uint64_t> cudaEdgesPtr(cudaEdges);
  thrust::copy_if(cudaRawEdgesPtr, cudaRawEdgesPtr + rawEdgesNum, cudaEdgesPtr, MoreThanCore(rawCorePtr, startK));

  edges = (uint64_t *)MyMalloc(edgesNum * sizeof(uint64_t));
  CUDA_TRY(cudaMemcpy(edges, cudaEdges, edgesNum * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaDeviceSynchronize());

  return edgesNum;
}

struct LessThan : public thrust::unary_function<uint64_t, bool> {
  __host__ __device__ bool operator()(uint64_t edge) { return FIRST(edge) < SECOND(edge); }
};

// 构造有向图
void ConstructHalfEdges(const uint64_t *edges, uint64_t *&halfEdges, EdgeT halfEdgesNum) {
  uint64_t *cudaEdges;
  uint64_t *cudaHalfEdges;
  CUDA_TRY(cudaMalloc((void **)&cudaEdges, halfEdgesNum * 2 * sizeof(uint64_t)));
  CUDA_TRY(cudaMemcpy(cudaEdges, edges, halfEdgesNum * 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

  CUDA_TRY(cudaMalloc((void **)&cudaHalfEdges, halfEdgesNum * sizeof(uint64_t)));

  thrust::device_ptr<uint64_t> cudaEdgesPtr(cudaEdges);
  thrust::device_ptr<uint64_t> cudaHalfEdgesPtr(cudaHalfEdges);
  thrust::copy_if(cudaEdgesPtr, cudaEdgesPtr + halfEdgesNum * 2, cudaHalfEdgesPtr, LessThan());

  halfEdges = (uint64_t *)MyMalloc(halfEdgesNum * sizeof(uint64_t));

  CUDA_TRY(cudaMemcpy(halfEdges, cudaHalfEdges, halfEdgesNum * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaDeviceSynchronize());

  CUDA_TRY(cudaFree(cudaEdges));
  CUDA_TRY(cudaFree(cudaHalfEdges));
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

__global__ void UnzipEdgesKernel(const uint64_t *edges, EdgeT edgesNum, NodeT *edgesFirst, NodeT *adj) {
  auto from = blockDim.x * blockIdx.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;
  for (EdgeT i = from; i < edgesNum; i += step) {
    uint64_t tmp = edges[i];
    edgesFirst[i] = FIRST(tmp);
    adj[i] = SECOND(tmp);
  }
}

__global__ void ConstructNodeIndexKernel(const NodeT *edgesFirst, EdgeT edgesNum, EdgeT *nodeIndex, NodeT nodesNum) {
  auto from = blockDim.x * blockIdx.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;
  for (uint64_t i = from; i <= edgesNum; i += step) {
    int64_t prev = i > 0 ? (int64_t)(edgesFirst[i - 1]) : -1;
    int64_t next = i < edgesNum ? (int64_t)(edgesFirst[i]) : nodesNum;
    for (int64_t j = prev + 1; j <= next; ++j) {
      nodeIndex[j] = i;
    }
  }
}

// 构建CSR
void ConstructCSRGraph(const uint64_t *edges, EdgeT edgesNum, EdgeT *&nodeIndex, NodeT *&adj) {
  uint64_t *cudaEdges;
  CUDA_TRY(cudaMalloc((void **)&cudaEdges, edgesNum * sizeof(uint64_t)));
  CUDA_TRY(cudaMemcpy(cudaEdges, edges, edgesNum * sizeof(uint64_t), cudaMemcpyHostToDevice));

  NodeT nodesNum = FIRST(edges[edgesNum - 1]) + 1;

  NodeT *cudaEdgesFirst;
  NodeT *cudaAdj;
  CUDA_TRY(cudaMalloc((void **)&cudaEdgesFirst, edgesNum * sizeof(NodeT)));
  CUDA_TRY(cudaMalloc((void **)&cudaAdj, edgesNum * sizeof(NodeT)));
  UnzipEdgesKernel<<<(edgesNum + 127) / 128, 128>>>(cudaEdges, edgesNum, cudaEdgesFirst, cudaAdj);
  CUDA_TRY(cudaDeviceSynchronize());

  EdgeT *cudaNodeIndex;
  CUDA_TRY(cudaMalloc((void **)&cudaNodeIndex, (nodesNum + 1) * sizeof(EdgeT)));
  ConstructNodeIndexKernel<<<(nodesNum + 127) / 128, 128>>>(cudaEdgesFirst, edgesNum, cudaNodeIndex, nodesNum);
  CUDA_TRY(cudaDeviceSynchronize());

  adj = (NodeT *)MyMalloc(edgesNum * sizeof(NodeT));
  nodeIndex = (EdgeT *)MyMalloc((nodesNum + 1) * sizeof(EdgeT));

  CUDA_TRY(cudaMemcpy(adj, cudaAdj, edgesNum * sizeof(NodeT), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaMemcpy(nodeIndex, cudaNodeIndex, (nodesNum + 1) * sizeof(EdgeT), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaDeviceSynchronize());

  CUDA_TRY(cudaFree(cudaEdges));
  CUDA_TRY(cudaFree(cudaEdgesFirst));
  CUDA_TRY(cudaFree(cudaAdj));
  CUDA_TRY(cudaFree(cudaNodeIndex));
}

//__inline__ __device__ EdgeT lowerBound(const NodeT *arr, NodeT start, NodeT end, NodeT value) {
//  while (start <= end) {
//    int middle = start + (end - start) / 2;
//    if (value > arr[middle]) {
//      start = middle + 1;
//    } else if (value < arr[middle]) {
//      end = middle - 1;
//    } else
//      return middle;
//  }
//  return 0;
//}
//
//__global__ void CUDAGetEdgesId(const uint64_t *edges, EdgeT edgesNum, const EdgeT *halfNodeIndex, const NodeT
//*halfAdj,
//                               EdgeT *edgesId) {
//  auto from = blockDim.x * blockIdx.x + threadIdx.x;
//  auto step = gridDim.x * blockDim.x;
//  for (EdgeT i = from; i < edgesNum; i += step) {
//    NodeT u = FIRST(edges[i]);
//    NodeT v = SECOND(edges[i]);
//    if (u > v) {
//      NodeT tmp = u;
//      u = v;
//      v = tmp;
//    }
//    edgesId[i] = lowerBound(halfAdj, halfNodeIndex[u], halfNodeIndex[u + 1], v);
//  }
//}
//
//// 边编号
// void GetEdgesId(const uint64_t *edges, EdgeT edgesNum, const EdgeT *halfNodeIndex, const NodeT *halfAdj,
//                EdgeT *&edgesId) {
//  CUDA_TRY(cudaMalloc((void **)&edgesId, edgesNum * sizeof(EdgeT)));
//  CUDA_TRY(cudaDeviceSynchronize());
//  log_info("1");
//  CUDAGetEdgesId<<<(edgesNum + 127) / 128, 128>>>(edges, edgesNum, halfNodeIndex, halfAdj, edgesId);
//  CUDA_TRY(cudaDeviceSynchronize());
//  log_info("2");
//}

// 边编号
void GetEdgesId(const uint64_t *edges, EdgeT edgesNum, const EdgeT *halfNodeIndex, const NodeT *halfAdj,
                EdgeT *&edgesId) {
  edgesId = (EdgeT *)MyMalloc(edgesNum * sizeof(EdgeT));

#pragma omp parallel for schedule(dynamic, 1024)
  for (EdgeT i = 0u; i < edgesNum; i++) {
    NodeT u = std::min(FIRST(edges[i]), SECOND(edges[i]));
    NodeT v = std::max(FIRST(edges[i]), SECOND(edges[i]));
    edgesId[i] = std::lower_bound(halfAdj + halfNodeIndex[u], halfAdj + halfNodeIndex[u + 1], v) - halfAdj;
  }
}
