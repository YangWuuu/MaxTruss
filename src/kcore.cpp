#include <algorithm>

#include "util.h"

#pragma ide diagnostic ignored "openmp-use-default-none"

// 并行扫描度取值
void Scan(NodeT nodesNum, const NodeT *core, NodeT level, NodeT *curr, NodeT &currTail) {
  NodeT buff[BUFFER_SIZE];
  NodeT index = 0;

#pragma omp for
  for (NodeT i = 0; i < nodesNum; i++) {
    if (core[i] == level) {
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
void SubLevel(const EdgeT *nodeIndex, const NodeT *adj, const NodeT *curr, NodeT currTail, NodeT *core, NodeT level,
              NodeT *next, NodeT &nextTail) {
  NodeT buff[BUFFER_SIZE];
  NodeT index = 0;

#pragma omp for
  for (NodeT i = 0; i < currTail; i++) {
    NodeT v = curr[i];
    for (EdgeT j = nodeIndex[v]; j < nodeIndex[v + 1]; j++) {
      NodeT u = adj[j];
      NodeT degU = core[u];

      if (degU > level) {
        NodeT du = __sync_fetch_and_sub(&core[u], 1);
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
    if (core[u] != level) {
      core[u] = level;
    }
  }
#pragma omp barrier
}

// 求解k-core的主流程
NodeT KCore(const EdgeT *nodeIndex, const NodeT *adj, NodeT nodesNum, NodeT *&core) {
  core = (NodeT *)MyMalloc(nodesNum * sizeof(NodeT));

#pragma omp parallel for
  for (NodeT i = 0; i < nodesNum; i++) {
    core[i] = nodeIndex[i + 1] - nodeIndex[i];
  }

  auto *curr = (NodeT *)MyMalloc(nodesNum * sizeof(NodeT));
  auto *next = (NodeT *)MyMalloc(nodesNum * sizeof(NodeT));
  NodeT currTail = 0;
  NodeT nextTail = 0;

#pragma omp parallel
  {
    NodeT todo = nodesNum;
    NodeT level = 0;

    while (todo > 0) {
      Scan(nodesNum, core, level, curr, currTail);
#pragma omp single
      { log_debug("level: %u currTail: %u restNodes: %u", level, currTail, todo); }
      while (currTail > 0) {
        todo = todo - currTail;
        SubLevel(nodeIndex, adj, curr, currTail, core, level, next, nextTail);
#pragma omp single
        {
          std::swap(curr, next);

          currTail = nextTail;
          nextTail = 0;
          log_debug("level: %u currTail: %u restNodes: %u", level, currTail, todo);
        }
#pragma omp barrier
      }
      level = level + 1;
#pragma omp barrier
    }
  }

  MyFree((void *&)curr, nodesNum * sizeof(NodeT));
  MyFree((void *&)next, nodesNum * sizeof(NodeT));

  NodeT maxCoreNum = 0;
#pragma omp parallel for reduction(max : maxCoreNum)
  for (NodeT i = 0; i < nodesNum; i++) {
    maxCoreNum = std::max(maxCoreNum, core[i]);
  }
  return maxCoreNum;
}
