#include "util.h"

const NodeT BITMAP_SCALE_LOG = 8u;
const NodeT BITMAP_SCALE = 1u << BITMAP_SCALE_LOG;
const NodeT TRI_BLOCK_SIZE = 1024u;
const NodeT WARP_SIZE = 1u << 5u;
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
                          NodeT *nodesCount, NodeT smBlocks, const EdgeT *edgesId, NodeT *edgesSup) {
  auto tid = threadIdx.x + blockDim.x * threadIdx.y;
  auto tnum = blockDim.x * blockDim.y;
  auto num_nodes = gridDim.x;
  const NodeT bitmapWordsNum = (num_nodes + ELEMENT_BITS - 1) / ELEMENT_BITS;
  const NodeT bitmapWordsNumIdx = DIV_ROUND_UP(bitmapWordsNum, BITMAP_SCALE);

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
  for (NodeT idx = tid; idx < bitmapWordsNumIdx; idx += tnum) {
    bitmapIndexes[idx] = 0;
  }
  __syncthreads();

  NodeT *bitmap = &bitmaps[bitmapWordsNum * (smBlocks * smId + bitmapIdx)];

  for (NodeT vIdx = uStart + tid; vIdx < uEnd; vIdx += tnum) {
    NodeT v = adj[vIdx];
    const NodeT vValue = v / ELEMENT_BITS;
    atomicOr(&bitmap[vValue], (0b1 << (v & (ELEMENT_BITS - 1))));
    atomicOr(&bitmapIndexes[vValue >> BITMAP_SCALE_LOG], (0b1 << ((v >> BITMAP_SCALE_LOG) & (ELEMENT_BITS - 1))));
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
      if ((bitmapIndexes[wVal >> BITMAP_SCALE_LOG] >> ((w >> BITMAP_SCALE_LOG) & (ELEMENT_BITS - 1))) & 0b1 == 1)
        if ((bitmap[wVal] >> (w & (ELEMENT_BITS - 1))) & 0b1 == 1) {
          count++;
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
  NodeT smNum = prop.multiProcessorCount;
  auto smBlocks = smThreads / TRI_BLOCK_SIZE;
  const NodeT bitmapWordsNum = (nodesNum + ELEMENT_BITS - 1) / ELEMENT_BITS;

  NodeT *nodesCount;
  NodeT *bitmaps;
  NodeT *bitmapStates;
  CUDA_TRY(cudaMallocManaged((void **)&bitmaps, smBlocks * smNum * bitmapWordsNum * sizeof(NodeT)));
  cudaMemset(bitmaps, 0, smBlocks * smNum * bitmapWordsNum * sizeof(NodeT));
  CUDA_TRY(cudaMallocManaged((void **)&bitmapStates, smBlocks * smNum * sizeof(NodeT)));
  cudaMemset(bitmapStates, 0, smBlocks * smNum * sizeof(NodeT));
  CUDA_TRY(cudaMallocManaged((void **)&nodesCount, sizeof(NodeT)));
  cudaMemset(nodesCount, 0, sizeof(NodeT));

  const NodeT bitmapWordsNumIdx = DIV_ROUND_UP(bitmapWordsNum, BITMAP_SCALE);
  BmpKernel<<<nodesNum, dim3(WARP_SIZE, TRI_BLOCK_SIZE / WARP_SIZE), bitmapWordsNumIdx * sizeof(NodeT)>>>(
      nodeIndex, adj, bitmaps, bitmapStates, nodesCount, smBlocks, edgesId, edgesSup);
  CUDA_TRY(cudaDeviceSynchronize());

  CUDA_TRY(cudaFree(bitmaps));
  CUDA_TRY(cudaFree(bitmapStates));
  CUDA_TRY(cudaFree(nodesCount));
}
