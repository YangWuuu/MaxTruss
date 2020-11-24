#include <algorithm>
#include <cstdlib>

#include "util.h"

#pragma ide diagnostic ignored "openmp-use-default-none"

// 并行扫描度取值
void Scan(NodeT n, const NodeT *deg, NodeT level, NodeT *curr,
          NodeT &currTail) {
  NodeT buff[BUFFER_SIZE];
  NodeT index = 0;

#pragma omp for
  for (NodeT i = 0; i < n; i++) {
    if (deg[i] == level) {
      buff[index] = i;
      index++;

      if (index >= BUFFER_SIZE) {
        NodeT tempIdx = __sync_fetch_and_add(&currTail, BUFFER_SIZE);
        for (NodeT j = 0; j < BUFFER_SIZE; j++) {
          curr[tempIdx + j] = buff[j];
        }
        index = 0;
      }
    }
  }

  if (index > 0) {
    NodeT tempIdx = __sync_fetch_and_add(&currTail, index);
    for (NodeT j = 0; j < index; j++) curr[tempIdx + j] = buff[j];
  }
#pragma omp barrier
}

// 子任务循环迭代分解
void SubLevel(const EdgeT *nodeIndex, const NodeT *edgesSecond,
              const NodeT *curr, NodeT currTail, NodeT *deg, NodeT level,
              NodeT *next, NodeT &nextTail) {
  NodeT buff[BUFFER_SIZE];
  NodeT index = 0;

#pragma omp for
  for (NodeT i = 0; i < currTail; i++) {
    NodeT v = curr[i];
    for (EdgeT j = nodeIndex[v]; j < nodeIndex[v + 1]; j++) {
      NodeT u = edgesSecond[j];
      NodeT degU = deg[u];

      if (degU > level) {
        NodeT du = __sync_fetch_and_sub(&deg[u], 1);
        if (du == (level + 1)) {
          buff[index] = u;
          index++;

          if (index >= BUFFER_SIZE) {
            NodeT tempIdx = __sync_fetch_and_add(&nextTail, BUFFER_SIZE);
            for (NodeT bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++) {
              next[tempIdx + bufIdx] = buff[bufIdx];
            }
            index = 0;
          }
        }
      }
    }
  }

  if (index > 0) {
    NodeT tempIdx = __sync_fetch_and_add(&nextTail, index);
    for (NodeT bufIdx = 0; bufIdx < index; bufIdx++) {
      next[tempIdx + bufIdx] = buff[bufIdx];
    }
  }
#pragma omp barrier

#pragma omp for
  for (NodeT i = 0; i < nextTail; i++) {
    NodeT u = next[i];
    if (deg[u] != level) {
      deg[u] = level;
    }
  }
#pragma omp barrier
}

// 求解k-core的主流程
void KCore(const EdgeT *nodeIndex, const NodeT *edgesSecond, NodeT nodesNum,
           NodeT *deg) {
  auto *curr = (NodeT *)myMalloc(nodesNum * sizeof(NodeT));
  auto *next = (NodeT *)myMalloc(nodesNum * sizeof(NodeT));
  NodeT currTail = 0;
  NodeT nextTail = 0;

#pragma omp parallel
  {
    NodeT todo = nodesNum;
    NodeT level = 0;

    while (todo > 0) {
      Scan(nodesNum, deg, level, curr, currTail);
      while (currTail > 0) {
        todo = todo - currTail;
        SubLevel(nodeIndex, edgesSecond, curr, currTail, deg, level, next,
                 nextTail);
#pragma omp single
        {
          std::swap(curr, next);

          currTail = nextTail;
          nextTail = 0;
        }
#pragma omp barrier
      }
      level = level + 1;
#pragma omp barrier
    }
  }
}
