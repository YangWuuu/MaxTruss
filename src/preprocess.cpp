#include <algorithm>
#include <cstdlib>

#include "util.h"

#pragma ide diagnostic ignored "openmp-use-default-none"

// 计算节点的度
void CalDeg(const uint64_t *edges, EdgeT edgesNum, NodeT nodesNum,
            NodeT *&deg) {
  deg = (NodeT *)calloc(nodesNum, sizeof(NodeT));
#ifdef SERIAL
  for (EdgeT i = 0; i < edgesNum; i++) {
    ++deg[FIRST(edges[i])];
  }
#else
#pragma omp parallel for
  for (EdgeT i = 0; i < edgesNum; i++) {
    __sync_fetch_and_add(&deg[FIRST(edges[i])], 1);
  }
#endif
}

// 边的解压缩
void Unzip(const uint64_t *edges, EdgeT edgesNum, NodeT *&edgesFirst,
           NodeT *&edgesSecond) {
  edgesFirst = (NodeT *)myMalloc(edgesNum * sizeof(NodeT));
  edgesSecond = (NodeT *)myMalloc(edgesNum * sizeof(NodeT));

#pragma omp parallel for
  for (EdgeT i = 0; i < edgesNum; i++) {
    edgesFirst[i] = FIRST(edges[i]);
    edgesSecond[i] = SECOND(edges[i]);
  }
}

// 转换CSR格式
void NodeIndex(const NodeT *deg, NodeT nodesNum, EdgeT *&nodeIndex) {
  nodeIndex = (EdgeT *)calloc((nodesNum + 1), sizeof(EdgeT));
  // 这里并行不一定比串行快，涉及到伪共享问题
  for (NodeT i = 0; i < nodesNum; i++) {
    nodeIndex[i + 1] = nodeIndex[i] + deg[i];
  }
}

// 边编号
void GetEdgesId(const uint64_t *edges, EdgeT edgesNum, EdgeT *&edgesId,
                const EdgeT *halfNodeIndex, const NodeT *halfEdgesSecond) {
  edgesId = (EdgeT *)myMalloc(edgesNum * sizeof(EdgeT));

#pragma omp parallel for schedule(dynamic, 1024)
  for (EdgeT i = 0u; i < edgesNum; i++) {
    NodeT u = std::min(FIRST(edges[i]), SECOND(edges[i]));
    NodeT v = std::max(FIRST(edges[i]), SECOND(edges[i]));
    edgesId[i] = std::lower_bound(halfEdgesSecond + halfNodeIndex[u],
                                  halfEdgesSecond + halfNodeIndex[u + 1], v) -
                 halfEdgesSecond;
  }
}
