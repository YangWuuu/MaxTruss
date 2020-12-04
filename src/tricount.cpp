#include "util.h"

#pragma ide diagnostic ignored "openmp-use-default-none"

// 三角形计数获取支持边数量
void GetEdgeSup(const EdgeT *halfNodeIndex, const NodeT *halfAdj, NodeT halfNodesNum, NodeT *&edgesSup) {
  EdgeT halfEdgesNum = halfNodeIndex[halfNodesNum];
  edgesSup = (NodeT *)MyMalloc(halfEdgesNum * sizeof(NodeT));

  auto *halfEdgesFirst = (NodeT *)MyMalloc(halfEdgesNum * sizeof(NodeT));

#pragma omp parallel for
  for (EdgeT i = 0; i < halfNodesNum; i++) {
    for (EdgeT j = halfNodeIndex[i]; j < halfNodeIndex[i + 1]; ++j) {
      halfEdgesFirst[j] = i;
    }
  }

#pragma omp parallel for schedule(dynamic, 1024)
  for (EdgeT i = 0; i < halfEdgesNum; i++) {
    NodeT u = halfEdgesFirst[i];
    NodeT v = halfAdj[i];
    if (v >= halfNodesNum) {
      continue;
    }
    EdgeT uStart = halfNodeIndex[u];
    EdgeT uEnd = halfNodeIndex[u + 1];
    EdgeT vStart = halfNodeIndex[v];
    EdgeT vEnd = halfNodeIndex[v + 1];
    while (uStart < uEnd && vStart < vEnd) {
      if (halfAdj[uStart] < halfAdj[vStart]) {
        ++uStart;
      } else if (halfAdj[uStart] > halfAdj[vStart]) {
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

  MyFree((void *&)halfEdgesFirst, halfEdgesNum * sizeof(NodeT));
}
