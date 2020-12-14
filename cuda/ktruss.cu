#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "log.h"
#include "util.h"

// 扫描支持边是否与truss层次相同
__global__ void ScanKernel(EdgeT halfEdgesNum, const NodeT *edgesSup, NodeT level, EdgeT *curr, EdgeT *currTail,
                           bool *inCurr) {
  auto from = blockDim.x * blockIdx.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;
  for (EdgeT i = from; i < halfEdgesNum; i += step) {
    if (edgesSup[i] == level) {
      inCurr[i] = true;
      curr[atomicAdd(currTail, 1)] = i;
    }
  }
}

// 扫描支持边层次小于指定层次
__global__ void ScanLessThanLevelKernel(EdgeT halfEdgesNum, const NodeT *edgesSup, NodeT level, EdgeT *curr,
                                        EdgeT *currTail, bool *inCurr) {
  auto from = blockDim.x * blockIdx.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;
  for (EdgeT i = from; i < halfEdgesNum; i += step) {
    if (edgesSup[i] <= level) {
      inCurr[i] = true;
      curr[atomicAdd(currTail, 1)] = i;
    }
  }
}

// 更新支持边的数值
__inline__ __device__ void UpdateSup(EdgeT e, NodeT *edgesSup, NodeT level, EdgeT *next, bool *inNext,
                                     EdgeT *nextTail) {
  NodeT supE = atomicSub(&edgesSup[e], 1);
  if (supE == (level + 1)) {
    auto insertIdx = atomicAdd(nextTail, 1);
    next[insertIdx] = e;
    inNext[e] = true;
  }
  if (supE <= level) {
    atomicAdd(&edgesSup[e], 1);
  }
}

__global__ void UpdateProcessKernel(const EdgeT *curr, const EdgeT currTail, bool *inCurr, bool *processed) {
  auto from = blockDim.x * blockIdx.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;
  for (EdgeT i = from; i < currTail; i += step) {
    EdgeT e = curr[i];
    processed[e] = true;
    inCurr[e] = false;
  }
}

template <typename T>
__host__ __device__ void swap(T &a, T &b) {
  T temp = a;
  a = b;
  b = temp;
}

__device__ EdgeT BinarySearch(NodeT target, const NodeT *adj, EdgeT start, EdgeT end) {
  EdgeT last = end;
  while (start < end) {
    EdgeT mid = start + ((end - start) >> 1u);
    if (adj[mid] < target) {
      start = mid + 1;
    } else if (adj[mid] > target) {
      end = mid;
    } else {
      return mid;
    }
  }
  return last;
}

__inline__ __device__ void PeelTriangle(NodeT level, const bool *inCurr, EdgeT *next, EdgeT *nextTail, bool *inNext,
                                        NodeT *edgesSup, bool *processed, EdgeT ee1, EdgeT ee2, EdgeT ee3) {
  if (processed[ee2] || processed[ee3]) {
    return;
  }
  if (edgesSup[ee2] > level && edgesSup[ee3] > level) {
    UpdateSup(ee2, edgesSup, level, next, inNext, nextTail);
    UpdateSup(ee3, edgesSup, level, next, inNext, nextTail);
  } else if (edgesSup[ee2] > level) {
    if ((ee1 < ee3 && inCurr[ee3]) || !inCurr[ee3]) {
      UpdateSup(ee2, edgesSup, level, next, inNext, nextTail);
    }
  } else if (edgesSup[ee3] > level) {
    if ((ee1 < ee2 && inCurr[ee2]) || !inCurr[ee2]) {
      UpdateSup(ee3, edgesSup, level, next, inNext, nextTail);
    }
  }
}

__global__ void SubLevelKernel(const EdgeT *nodeIndex, const NodeT *adj, const EdgeT *curr, bool *inCurr,
                               EdgeT currTail, NodeT *edgesSup, NodeT level, EdgeT *next, bool *inNext, EdgeT *nextTail,
                               bool *processed, const EdgeT *edgesId, const uint64_t *halfEdges) {
  __shared__ EdgeT size;
  extern __shared__ EdgeT shared[];
  EdgeT *eArr1 = shared;
  EdgeT *eArr2 = shared + blockDim.x * 2;
  EdgeT *eArr3 = shared + blockDim.x * 2 * 2;
  if (threadIdx.x == 0) {
    size = 0;
  }
  __syncthreads();

  for (EdgeT i = blockIdx.x; i < currTail; i += gridDim.x) {
    EdgeT e1 = curr[i];
    NodeT u = FIRST(halfEdges[e1]);
    NodeT v = SECOND(halfEdges[e1]);

    EdgeT uStart = nodeIndex[u];
    EdgeT uEnd = nodeIndex[u + 1];
    EdgeT vStart = nodeIndex[v];
    EdgeT vEnd = nodeIndex[v + 1];

    if (uEnd - uStart > vEnd - vStart) {
      swap(u, v);
      swap(uStart, vStart);
      swap(uEnd, vEnd);
    }

    for (auto e2 = uStart + threadIdx.x; e2 < uStart + DIV_ROUND_UP(uEnd - uStart, blockDim.x) * blockDim.x;
         e2 += blockDim.x) {
      __syncthreads();

      if (size >= blockDim.x) {
        for (EdgeT j = threadIdx.x; j < size; j += blockDim.x) {
          EdgeT ee1 = eArr1[j];
          EdgeT ee2 = edgesId[eArr2[j]];
          EdgeT ee3 = edgesId[eArr3[j]];
          PeelTriangle(level, inCurr, next, nextTail, inNext, edgesSup, processed, ee1, ee2, ee3);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          size = 0;
        }
        __syncthreads();
      }

      EdgeT e3 = vEnd;
      if (e2 < uEnd) {
        e3 = BinarySearch(adj[e2], adj, vStart, vEnd);
      }
      if (e3 != vEnd) {
        auto pos = atomicAdd(&size, 1);
        eArr1[pos] = e1;
        eArr2[pos] = e2;
        eArr3[pos] = e3;
      }
      __syncthreads();
    }
  }
  __syncthreads();
  for (EdgeT j = threadIdx.x; j < size; j += blockDim.x) {
    EdgeT ee1 = eArr1[j];
    EdgeT ee2 = edgesId[eArr2[j]];
    EdgeT ee3 = edgesId[eArr3[j]];
    PeelTriangle(level, inCurr, next, nextTail, inNext, edgesSup, processed, ee1, ee2, ee3);
  }
}

// 子任务循环迭代消减truss
void SubLevel(const EdgeT *nodeIndex, const NodeT *adj, const EdgeT *curr, bool *inCurr, EdgeT *currTail,
              NodeT *edgesSup, NodeT level, EdgeT *next, bool *inNext, EdgeT *nextTail, bool *processed,
              const EdgeT *edgesId, const uint64_t *halfEdges) {
  SubLevelKernel<<<*currTail, BLOCK_SIZE, BLOCK_SIZE * sizeof(EdgeT) * 2 * 3>>>(
      nodeIndex, adj, curr, inCurr, *currTail, edgesSup, level, next, inNext, nextTail, processed, edgesId, halfEdges);
  CUDA_TRY(cudaDeviceSynchronize());

  UpdateProcessKernel<<<DIV_ROUND_UP(*currTail, BLOCK_SIZE), BLOCK_SIZE>>>(curr, *currTail, inCurr, processed);
  CUDA_TRY(cudaDeviceSynchronize());
}

// 获取各层次truss的边的数量
NodeT DisplayStats(const NodeT *edgesSup, EdgeT halfEdgesNum, NodeT minK) {
  thrust::device_ptr<NodeT> edgesSupPtr(const_cast<NodeT *>(edgesSup));
  NodeT maxSup = *thrust::max_element(edgesSupPtr, edgesSupPtr + halfEdgesNum);

  EdgeT numEdgesWithMaxSup = thrust::count(edgesSupPtr, edgesSupPtr + halfEdgesNum, maxSup);

  log_info("Max-truss: %u  Edges in Max-truss: %u", maxSup + 2, numEdgesWithMaxSup);
  if (maxSup + 2 >= minK) {
    printf("kmax = %u, Edges in kmax-truss = %u.\n", maxSup + 2, numEdgesWithMaxSup);
  }
  return maxSup + 2;
}

void InitCuda(EdgeT *&currTail, EdgeT *&nextTail, bool *&processed, bool *&inCurr, bool *&inNext, EdgeT *&curr,
              EdgeT *&next, const EdgeT halfEdgesNum) {
  CUDA_TRY(cudaMallocManaged((void **)&currTail, sizeof(EdgeT)));
  CUDA_TRY(cudaMallocManaged((void **)&nextTail, sizeof(EdgeT)));
  *currTail = 0;
  *nextTail = 0;

  CUDA_TRY(cudaMallocManaged((void **)&processed, halfEdgesNum * sizeof(bool)));
  CUDA_TRY(cudaMallocManaged((void **)&inCurr, halfEdgesNum * sizeof(bool)));
  CUDA_TRY(cudaMallocManaged((void **)&inNext, halfEdgesNum * sizeof(bool)));
  CUDA_TRY(cudaMallocManaged((void **)&curr, halfEdgesNum * sizeof(EdgeT)));
  CUDA_TRY(cudaMallocManaged((void **)&next, halfEdgesNum * sizeof(EdgeT)));

  CUDA_TRY(cudaMemset(processed, 0, halfEdgesNum * sizeof(bool)));
  CUDA_TRY(cudaMemset(inCurr, 0, halfEdgesNum * sizeof(bool)));
  CUDA_TRY(cudaMemset(inNext, 0, halfEdgesNum * sizeof(bool)));
}

__global__ void DetectDeletedEdgesKernel(EdgeT *nodeIndex, EdgeT *edgesId, bool *processed, NodeT nodesNum,
                                         EdgeT *newOffsets, bool *edgesDeleted) {
  __shared__ NodeT cnts[WARPS_PER_BLOCK];

  auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
  auto gtnum = blockDim.x * gridDim.x;
  auto gwid = gtid >> WARP_BITS;
  auto gwnum = gtnum >> WARP_BITS;
  auto lane = threadIdx.x & WARP_MASK;
  auto lwid = threadIdx.x >> WARP_BITS;

  for (auto u = gwid; u < nodesNum; u += gwnum) {
    if (0 == lane) {
      cnts[lwid] = 0;
    }
    __syncwarp();

    auto start = nodeIndex[u];
    auto end = nodeIndex[u + 1];
    for (auto v_idx = start + lane; v_idx < end; v_idx += WARP_SIZE) {
      auto target_edge_idx = edgesId[v_idx];
      edgesDeleted[v_idx] = !processed[target_edge_idx];
      if (edgesDeleted[v_idx]) {
        atomicAdd(&cnts[lwid], 1);
      }
    }
    __syncwarp();

    if (0 == lane) {
      newOffsets[u] = cnts[lwid];
    }
  }
}

template <typename T>
struct IsDelete : public thrust::unary_function<T, bool> {
  template <typename Tuple>
  __host__ __device__ bool operator()(const Tuple &tuple) {
    auto y = thrust::get<1>(tuple);
    return y;
  }
};

void ShrinkGraph(EdgeT *&nodeIndex, NodeT *&adj, EdgeT *&edgesId, bool *&processed, NodeT nodesNum,
                 EdgeT *&newNodeIndex, NodeT *&newAdj, EdgeT *&newEdgesId, bool *&edgesDeleted, EdgeT oldEdgesNum,
                 EdgeT newEdgesNum) {
  DetectDeletedEdgesKernel<<<GRID_SIZE, BLOCK_SIZE>>>(nodeIndex, edgesId, processed, nodesNum, newNodeIndex,
                                                      edgesDeleted);

  thrust::device_ptr<EdgeT> newNodeIndexPtr(newNodeIndex);
  thrust::exclusive_scan(newNodeIndexPtr, newNodeIndexPtr + nodesNum + 1, newNodeIndexPtr);

  swap(nodeIndex, newNodeIndex);

  thrust::device_ptr<NodeT> adjPtr(adj);
  thrust::device_ptr<NodeT> newAdjPtr(newAdj);
  thrust::device_ptr<EdgeT> edgesIdPtr(edgesId);
  thrust::device_ptr<EdgeT> newEdgesIdPtr(newEdgesId);
  thrust::device_ptr<bool> deleteEdgesPtr(edgesDeleted);

  thrust::copy_if(
      thrust::make_zip_iterator(thrust::make_tuple(adjPtr, deleteEdgesPtr)),
      thrust::make_zip_iterator(thrust::make_tuple(adjPtr + oldEdgesNum * 2, deleteEdgesPtr + oldEdgesNum * 2)),
      thrust::make_zip_iterator(thrust::make_tuple(newAdjPtr, thrust::make_discard_iterator())),
      IsDelete<decltype(thrust::make_tuple(adjPtr, deleteEdgesPtr))>());

  thrust::copy_if(
      thrust::make_zip_iterator(thrust::make_tuple(edgesIdPtr, deleteEdgesPtr)),
      thrust::make_zip_iterator(thrust::make_tuple(edgesIdPtr + oldEdgesNum * 2, deleteEdgesPtr + oldEdgesNum * 2)),
      thrust::make_zip_iterator(thrust::make_tuple(newEdgesIdPtr, thrust::make_discard_iterator())),
      IsDelete<decltype(thrust::make_tuple(edgesIdPtr, deleteEdgesPtr))>());

  swap(adj, newAdj);
  swap(edgesId, newEdgesId);
}

// 求解k-truss的主流程
void KTruss(EdgeT *nodeIndex, NodeT *adj, EdgeT *edgesId, NodeT nodesNum, const uint64_t *halfEdges, EdgeT halfEdgesNum,
            NodeT *edgesSup, NodeT startLevel) {
  EdgeT *currTail;
  EdgeT *nextTail;

  bool *processed;
  bool *inCurr;
  bool *inNext;
  EdgeT *curr;
  EdgeT *next;

  InitCuda(currTail, nextTail, processed, inCurr, inNext, curr, next, halfEdgesNum);

  NodeT *newAdj;
  EdgeT *newEdgesId;
  EdgeT *newNodeIndex;
  bool *edgesDeleted;

  CUDA_TRY(cudaMallocManaged((void **)&newAdj, halfEdgesNum * 2 * sizeof(NodeT)));
  CUDA_TRY(cudaMallocManaged((void **)&newEdgesId, halfEdgesNum * 2 * sizeof(EdgeT)));
  CUDA_TRY(cudaMallocManaged((void **)&newNodeIndex, (nodesNum + 1) * sizeof(EdgeT)));
  CUDA_TRY(cudaMallocManaged((void **)&edgesDeleted, halfEdgesNum * 2 * sizeof(bool)));

  NodeT level = startLevel;
  EdgeT todo = halfEdgesNum;
  EdgeT oriHalfEdgesNum = halfEdgesNum;
  EdgeT deleteEdgesNum = 0;
  if (level > 0u) {
    --level;
    ScanLessThanLevelKernel<<<DIV_ROUND_UP(oriHalfEdgesNum, BLOCK_SIZE), BLOCK_SIZE>>>(oriHalfEdgesNum, edgesSup, level,
                                                                                       curr, currTail, inCurr);
    CUDA_TRY(cudaDeviceSynchronize());
    log_debug("level: %u currTail: %u restEdges: %u", level, *currTail, todo);

    while (*currTail > 0) {
      if ((deleteEdgesNum * 1.0 / oriHalfEdgesNum) > 0.05) {
        log_debug("ShrinkGraph: %u %u", deleteEdgesNum, todo);
        ShrinkGraph(nodeIndex, adj, edgesId, processed, nodesNum, newNodeIndex, newAdj, newEdgesId, edgesDeleted,
                    halfEdgesNum, todo);
        halfEdgesNum = todo;
        deleteEdgesNum = 0;
      }
      todo = todo - *currTail;

      deleteEdgesNum += *currTail;

      SubLevel(nodeIndex, adj, curr, inCurr, currTail, edgesSup, level, next, inNext, nextTail, processed, edgesId,
               halfEdges);

      std::swap(curr, next);
      std::swap(inCurr, inNext);

      *currTail = *nextTail;
      *nextTail = 0;

      log_debug("level: %u currTail: %u restEdges: %u deleteEdgesNum: %u", level, *currTail, todo, deleteEdgesNum);
    }
    ++level;
  } else {
    thrust::device_ptr<EdgeT> edgesSupPtr(edgesSup);
    level = *thrust::min_element(edgesSupPtr, edgesSupPtr + halfEdgesNum);
  }

  while (todo > 0) {
    //    if ((deleteEdgesNum * 1.0 / oriHalfEdgesNum) > 0.05) {
    //      log_debug("ShrinkGraph: %u %u", deleteEdgesNum, todo);
    //      ShrinkGraph(nodeIndex, adj, edgesId, processed, nodesNum, newNodeIndex, newAdj, newEdgesId, edgesDeleted,
    //                  halfEdgesNum, todo);
    //      halfEdgesNum = todo;
    //      deleteEdgesNum = 0;
    //    }

    ScanKernel<<<DIV_ROUND_UP(oriHalfEdgesNum, BLOCK_SIZE), BLOCK_SIZE>>>(oriHalfEdgesNum, edgesSup, level, curr,
                                                                          currTail, inCurr);
    CUDA_TRY(cudaDeviceSynchronize());
    log_debug("level: %u currTail: %u restEdges: %u", level, *currTail, todo);

    while (*currTail > 0) {
      if ((deleteEdgesNum * 1.0 / oriHalfEdgesNum) > 0.1) {
        log_debug("ShrinkGraph: %u %u", deleteEdgesNum, todo);
        ShrinkGraph(nodeIndex, adj, edgesId, processed, nodesNum, newNodeIndex, newAdj, newEdgesId, edgesDeleted,
                    halfEdgesNum, todo);
        halfEdgesNum = todo;
        deleteEdgesNum = 0;
      }

      todo = todo - *currTail;

      deleteEdgesNum += *currTail;

      SubLevel(nodeIndex, adj, curr, inCurr, currTail, edgesSup, level, next, inNext, nextTail, processed, edgesId,
               halfEdges);

      std::swap(curr, next);
      std::swap(inCurr, inNext);

      *currTail = *nextTail;
      *nextTail = 0;

      log_debug("level: %u currTail: %u restEdges: %u deleteEdgesNum: %u", level, *currTail, todo, deleteEdgesNum);
    }
    ++level;
  }

  CUDA_TRY(cudaFree(currTail));
  CUDA_TRY(cudaFree(nextTail));
  CUDA_TRY(cudaFree(processed));
  CUDA_TRY(cudaFree(inCurr));
  CUDA_TRY(cudaFree(inNext));
  CUDA_TRY(cudaFree(curr));
  CUDA_TRY(cudaFree(next));
}
