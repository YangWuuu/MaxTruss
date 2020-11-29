#include <algorithm>
#include <cstdlib>

#include "util.h"

#pragma ide diagnostic ignored "openmp-use-default-none"

// 图的裁剪
EdgeT ConstructNewGraph(const uint64_t *rawEdges, uint64_t *&edges, const NodeT *rawCore, EdgeT rawEdgesNum,
                        NodeT startK) {
  edges = (uint64_t *)MyMalloc(rawEdgesNum * sizeof(uint64_t));
  // TODO parallel
  return std::copy_if(rawEdges, rawEdges + rawEdgesNum, edges,
                      [&](const uint64_t edge) {
                        return rawCore[FIRST(edge)] >= (startK - 2) && rawCore[SECOND(edge)] >= (startK - 2);
                      }) -
         edges;
}

// 构造有向图
void ConstructHalfEdges(const uint64_t *edges, uint64_t *&halfEdges, EdgeT halfEdgesNum) {
  halfEdges = (uint64_t *)MyMalloc(halfEdgesNum * sizeof(uint64_t));
  // TODO parallel
  std::copy_if(edges, edges + halfEdgesNum * 2, halfEdges,
               [](const uint64_t &edge) { return FIRST(edge) < SECOND(edge); });
}

// 构建CSR
void ConstructCSRGraph(const uint64_t *edges, EdgeT edgesNum, EdgeT *&nodeIndex, NodeT *&adj) {
  auto *edgesFirst = (NodeT *)MyMalloc(edgesNum * sizeof(NodeT));
  adj = (NodeT *)MyMalloc(edgesNum * sizeof(NodeT));

  NodeT nodesNum = FIRST(edges[edgesNum - 1]) + 1;
  nodeIndex = (EdgeT *)MyMalloc((nodesNum + 1) * sizeof(EdgeT));

#pragma omp parallel for
  for (EdgeT i = 0; i < edgesNum; i++) {
    edgesFirst[i] = FIRST(edges[i]);
    adj[i] = SECOND(edges[i]);
  }

#pragma omp parallel for
  for (EdgeT i = 0; i <= edgesNum; i++) {
    int64_t prev = i > 0 ? (int64_t)edgesFirst[i - 1] : -1;
    int64_t next = i < edgesNum ? (int64_t)edgesFirst[i] : nodesNum;
    for (int64_t j = prev + 1; j <= next; ++j) {
      nodeIndex[j] = i;
    }
  }

  MyFree((void *&)edgesFirst, edgesNum * sizeof(NodeT));
}

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
