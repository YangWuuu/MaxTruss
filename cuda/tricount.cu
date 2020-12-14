#include "util.h"

const NodeT ELEMENT_BITS = sizeof(NodeT) * 8;

#define WARP_REDUCE(var)                          \
  {                                               \
    var += __shfl_down_sync(0xFFFFFFFF, var, 16); \
    var += __shfl_down_sync(0xFFFFFFFF, var, 8);  \
    var += __shfl_down_sync(0xFFFFFFFF, var, 4);  \
    var += __shfl_down_sync(0xFFFFFFFF, var, 2);  \
    var += __shfl_down_sync(0xFFFFFFFF, var, 1);  \
  }

__global__ void BmpKernel(const EdgeT *nodeIndex, const NodeT *adj, NodeT *bitmaps, NodeT *bitmapStates,
                          NodeT bloomFilterSize, NodeT *nodesCount, NodeT smBlocks, const EdgeT *edgesId,
                          NodeT *edgesSup) {
  auto tid = threadIdx.x + blockDim.x * threadIdx.y;
  auto tnum = blockDim.x * blockDim.y;
  const NodeT bitmapSize = (gridDim.x + ELEMENT_BITS - 1) / ELEMENT_BITS;

  __shared__ NodeT u, uStart, uEnd;
  __shared__ NodeT smId, bitmapIdx, startBitmap, endBitmap;

  extern __shared__ NodeT bitmapIndexes[];

  if (tid == 0) {
    u = atomicAdd(nodesCount, 1);
    uStart = nodeIndex[u];
    uEnd = nodeIndex[u + 1];
    startBitmap = adj[uStart] / ELEMENT_BITS;
    endBitmap = (uStart == uEnd) ? startBitmap : adj[uEnd - 1] / ELEMENT_BITS;
  } else if (tid == tnum - 1) {
    NodeT temp = 0;
    asm("mov.u32 %0, %smid;" : "=r"(smId));
    /*get current SM*/
    while (atomicCAS(&bitmapStates[smId * smBlocks + temp], 0, 1) != 0) {
      temp++;
    }
    bitmapIdx = temp;
  }
  // initialize the 2-level bitmap
  for (NodeT idx = tid; idx < bloomFilterSize; idx += tnum) {
    bitmapIndexes[idx] = 0;
  }
  __syncthreads();

  NodeT *bitmap = &bitmaps[bitmapSize * (smBlocks * smId + bitmapIdx)];

  for (NodeT vIdx = uStart + tid; vIdx < uEnd; vIdx += tnum) {
    NodeT v = adj[vIdx];
    const NodeT vValue = v / ELEMENT_BITS;
    atomicOr(&bitmap[vValue], (0b1 << (v & (ELEMENT_BITS - 1))));
    atomicOr(&bitmapIndexes[vValue % bloomFilterSize], (0b1 << (v & (ELEMENT_BITS - 1))));
  }
  __syncthreads();

  // loop the neighbors
  // x dimension: warp-size
  // y dimension: number of warps
  auto du = nodeIndex[u + 1] - nodeIndex[u];
  for (NodeT vIdx = uStart + threadIdx.y; vIdx < uEnd; vIdx += blockDim.y) {
    // each warp processes a node
    NodeT count = 0;
    NodeT v = adj[vIdx];
    auto dv = nodeIndex[v + 1] - nodeIndex[v];
    if (dv > du || ((du == dv) && u > v)) {
      continue;
    }
    NodeT vStart = nodeIndex[v];
    NodeT vEnd = nodeIndex[v + 1];
    for (NodeT wIdx = vStart + threadIdx.x; wIdx < vEnd; wIdx += blockDim.x) {
      NodeT w = adj[wIdx];
      const NodeT wVal = w / ELEMENT_BITS;
      if ((bitmapIndexes[wVal % bloomFilterSize] >> (w & (ELEMENT_BITS - 1))) & 0b1 == 1) {
        if ((bitmap[wVal] >> (w & (ELEMENT_BITS - 1))) & 0b1 == 1) {
          count++;
        }
      }
    }
    __syncwarp();
    // warp-wise reduction
    WARP_REDUCE(count);
    if (threadIdx.x == 0) {
      edgesSup[edgesId[vIdx]] = count;
    }
  }
  __syncthreads();

  // clean the bitmap
  if (endBitmap - startBitmap + 1 <= uEnd - uStart) {
    for (NodeT idx = startBitmap + tid; idx <= endBitmap; idx += tnum) {
      bitmap[idx] = 0;
    }
  } else {
    for (NodeT vIdx = uStart + tid; vIdx < uEnd; vIdx += tnum) {
      NodeT v = adj[vIdx];
      bitmap[v / ELEMENT_BITS] = 0;
    }
  }
  __syncthreads();

  // release the bitmap lock
  if (tid == 0) {
    atomicCAS(&bitmapStates[smId * smBlocks + bitmapIdx], 1, 0);
  }
}

void GetEdgeSup(const EdgeT *nodeIndex, const NodeT *adj, const EdgeT *edgesId, NodeT nodesNum, NodeT *&edgesSup) {
  EdgeT edgesNum = nodeIndex[nodesNum];
  CUDA_TRY(cudaMallocManaged((void **)&edgesSup, edgesNum / 2 * sizeof(NodeT)));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  NodeT smThreads = prop.maxThreadsPerMultiProcessor;
  NodeT blockThreads = prop.maxThreadsPerBlock;
  NodeT smNum = prop.multiProcessorCount;
  NodeT sharedMem = prop.sharedMemPerMultiprocessor;
  auto smBlocks = smThreads / blockThreads;
  const NodeT bitmapSize = (nodesNum + ELEMENT_BITS - 1) / ELEMENT_BITS;
  const NodeT bloomFilterSize = sharedMem / sizeof(NodeT) / smBlocks - 8;
  log_info(
      "smThreads: %u blockThreads: %u smNum: %u smBlocks: %u bitmapSize: %u sharedMem: %u "
      "bloomFilterSize: %u",
      smThreads, blockThreads, smNum, smBlocks, bitmapSize, sharedMem, bloomFilterSize);

  NodeT *nodesCount;
  NodeT *bitmaps;
  NodeT *bitmapStates;
  CUDA_TRY(cudaMallocManaged((void **)&bitmaps, smBlocks * smNum * bitmapSize * sizeof(NodeT)));
  cudaMemset(bitmaps, 0, smBlocks * smNum * bitmapSize * sizeof(NodeT));
  CUDA_TRY(cudaMallocManaged((void **)&bitmapStates, smBlocks * smNum * sizeof(NodeT)));
  cudaMemset(bitmapStates, 0, smBlocks * smNum * sizeof(NodeT));
  CUDA_TRY(cudaMallocManaged((void **)&nodesCount, sizeof(NodeT)));
  cudaMemset(nodesCount, 0, sizeof(NodeT));

  BmpKernel<<<nodesNum, dim3(WARP_SIZE, blockThreads / WARP_SIZE), bloomFilterSize * sizeof(NodeT)>>>(
      nodeIndex, adj, bitmaps, bitmapStates, bloomFilterSize, nodesCount, smBlocks, edgesId, edgesSup);
  CUDA_TRY(cudaDeviceSynchronize());

  CUDA_TRY(cudaFree(bitmaps));
  CUDA_TRY(cudaFree(bitmapStates));
  CUDA_TRY(cudaFree(nodesCount));
}
