#include <cstdlib>

#include "util.h"

#pragma ide diagnostic ignored "openmp-use-default-none"

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
