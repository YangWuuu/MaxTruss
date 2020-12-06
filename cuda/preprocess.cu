#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>

#include <algorithm>
#include <cstdlib>

#include "util.h"

struct MoreThanCore : public thrust::unary_function<uint64_t, bool> {
  thrust::device_ptr<NodeT> rawCorePtr_;
  NodeT startK_;

  MoreThanCore(thrust::device_ptr<NodeT> rawCorePtr, NodeT startK) : rawCorePtr_(rawCorePtr), startK_(startK) {}
  __host__ __device__ bool operator()(uint64_t edge) {
    return rawCorePtr_[FIRST(edge)] >= (startK_ - 2) && rawCorePtr_[SECOND(edge)] >= (startK_ - 2);
  }
};

// 图的裁剪
EdgeT ConstructNewGraph(const uint64_t *rawEdges, uint64_t *&edges, const NodeT *rawCore, EdgeT rawEdgesNum,
                        NodeT startK) {
  // TODO rawEdges 可能需要分块

  thrust::device_ptr<uint64_t> rawEdgesPtr(const_cast<uint64_t *>(rawEdges));
  thrust::device_ptr<NodeT> rawCorePtr(const_cast<NodeT *>(rawCore));
  EdgeT edgesNum = thrust::count_if(rawEdgesPtr, rawEdgesPtr + rawEdgesNum, MoreThanCore(rawCorePtr, startK));

  CUDA_TRY(cudaMallocManaged((void **)&edges, edgesNum * sizeof(uint64_t)));
  thrust::device_ptr<uint64_t> edgesPtr(edges);
  thrust::copy_if(rawEdgesPtr, rawEdgesPtr + rawEdgesNum, edgesPtr, MoreThanCore(rawCorePtr, startK));

  return edgesNum;
}

struct LessThan : public thrust::unary_function<uint64_t, bool> {
  __host__ __device__ bool operator()(uint64_t edge) { return FIRST(edge) < SECOND(edge); }
};

// 构造有向图
void ConstructHalfEdges(const uint64_t *edges, uint64_t *&halfEdges, EdgeT halfEdgesNum) {
  CUDA_TRY(cudaMallocManaged((void **)&halfEdges, halfEdgesNum * sizeof(uint64_t)));

  thrust::device_ptr<uint64_t> edgesPtr(const_cast<uint64_t *>(edges));
  thrust::device_ptr<uint64_t> halfEdgesPtr(halfEdges);
  thrust::copy_if(edgesPtr, edgesPtr + halfEdgesNum * 2, halfEdgesPtr, LessThan());
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
  NodeT nodesNum = FIRST(edges[edgesNum - 1]) + 1;

  NodeT *edgesFirst;
  CUDA_TRY(cudaMallocManaged((void **)&edgesFirst, edgesNum * sizeof(NodeT)));
  CUDA_TRY(cudaMallocManaged((void **)&adj, edgesNum * sizeof(NodeT)));
  UnzipEdgesKernel<<<(edgesNum + 127) / 128, 128>>>(edges, edgesNum, edgesFirst, adj);
  CUDA_TRY(cudaDeviceSynchronize());

  CUDA_TRY(cudaMallocManaged((void **)&nodeIndex, (nodesNum + 1) * sizeof(EdgeT)));
  ConstructNodeIndexKernel<<<(nodesNum + 127) / 128, 128>>>(edgesFirst, edgesNum, nodeIndex, nodesNum);
  CUDA_TRY(cudaDeviceSynchronize());

  CUDA_TRY(cudaFree(edgesFirst));
}

__inline__ __device__ EdgeT lowerBound(const NodeT *arr, NodeT start, NodeT end, NodeT value) {
  while (start <= end) {
    int mid = start + (end - start) / 2;
    if (arr[mid] < value) {
      start = mid + 1;
    } else if (arr[mid] > value) {
      end = mid;
    } else
      return mid;
  }
  return start;
}

__global__ void CUDAGetEdgesId(const uint64_t *edges, EdgeT edgesNum, const EdgeT *halfNodeIndex, const NodeT *halfAdj,
                               EdgeT *edgesId) {
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
    edgesId[i] = lowerBound(halfAdj, halfNodeIndex[u], halfNodeIndex[u + 1], v);
  }
}

// 边编号
void GetEdgesId(const uint64_t *edges, EdgeT edgesNum, const EdgeT *halfNodeIndex, const NodeT *halfAdj,
                EdgeT *&edgesId) {
  CUDA_TRY(cudaMallocManaged((void **)&edgesId, edgesNum * sizeof(EdgeT)));
  CUDAGetEdgesId<<<(edgesNum + 127) / 128, 128>>>(edges, edgesNum, halfNodeIndex, halfAdj, edgesId);
  CUDA_TRY(cudaDeviceSynchronize());
}
