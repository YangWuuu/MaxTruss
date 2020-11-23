#include <cstdlib>

#include "util.h"

#pragma ide diagnostic ignored "openmp-use-default-none"

//__global__ void GPUCalculateTrianglesSmall(uint64_t m, uint32_t *edges_first,
//                                           uint32_t *edges_second,
//                                           uint64_t *nodes, uint64_t *results)
//                                           {
//  auto from = blockDim.x * blockIdx.x + threadIdx.x;
//  auto step = gridDim.x * blockDim.x;
//
//  uint64_t count = 0;
//  for (uint64_t i = from; i < m; i += step) {
//    uint32_t u = edges_second[i], v = edges_first[i];
//    uint32_t u_it = nodes[u], u_end = nodes[u + 1];
//    uint32_t v_it = nodes[v], v_end = nodes[v + 1];
//    int64_t a = edges_first[u_it], b = edges_first[v_it];
//    while (u_it != u_end && v_it != v_end) {
//      int64_t d = a - b;
//      if (d <= 0) a = edges_first[++u_it];
//      if (d >= 0) b = edges_first[++v_it];
//      if (d == 0) ++count;
//    }
//  }
//  results[blockDim.x * blockIdx.x + threadIdx.x] = count;
//}

// 三角形计数获取支持边数量
void GetEdgeSup(EdgeT halfEdgesNum, NodeT *&halfEdgesFirst,
                NodeT *&halfEdgesSecond, NodeT *&halfDeg, EdgeT *&halfNodeIndex,
                NodeT *&edgesSup) {
  edgesSup = (NodeT *)calloc(halfEdgesNum, sizeof(NodeT));
#pragma omp parallel for schedule(dynamic, 1024)
  for (EdgeT i = 0; i < halfEdgesNum; i++) {
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
        __sync_fetch_and_add(&edgesSup[i], 1);
        __sync_fetch_and_add(&edgesSup[uStart], 1);
        __sync_fetch_and_add(&edgesSup[vStart], 1);
        ++uStart;
        ++vStart;
      }
    }
  }
}
