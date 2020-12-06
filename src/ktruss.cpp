#include <algorithm>
#include <cstdlib>

#include "log.h"
#include "util.h"

#pragma ide diagnostic ignored "openmp-use-default-none"

// 并行扫描支持边是否与truss层次相同
void Scan(EdgeT edgesNum, const NodeT *edgesSup, NodeT level, EdgeT *curr, EdgeT &currTail, bool *inCurr) {
  NodeT buff[BUFFER_SIZE];
  EdgeT index = 0;

#pragma omp for
  for (EdgeT i = 0; i < edgesNum; i++) {
    if (edgesSup[i] == level) {
      buff[index] = i;
      inCurr[i] = true;
      index++;

      if (index >= BUFFER_SIZE) {
        EdgeT tempIdx = __sync_fetch_and_add(&currTail, BUFFER_SIZE);
        for (EdgeT j = 0; j < BUFFER_SIZE; j++) {
          curr[tempIdx + j] = buff[j];
        }
        index = 0;
      }
    }
  }

  if (index > 0) {
    EdgeT tempIdx = __sync_fetch_and_add(&currTail, index);
    for (EdgeT j = 0; j < index; j++) {
      curr[tempIdx + j] = buff[j];
    }
  }
#pragma omp barrier
}

// 并行扫描支持边层次小于指定层次
void ScanLessThanLevel(EdgeT edgesNum, const NodeT *edgesSup, NodeT level, EdgeT *curr, EdgeT &currTail, bool *inCurr) {
  NodeT buff[BUFFER_SIZE];
  EdgeT index = 0;

#pragma omp for
  for (EdgeT i = 0; i < edgesNum; i++) {
    if (edgesSup[i] <= level) {
      buff[index] = i;
      inCurr[i] = true;
      index++;

      if (index >= BUFFER_SIZE) {
        EdgeT tempIdx = __sync_fetch_and_add(&currTail, BUFFER_SIZE);
        for (EdgeT j = 0; j < BUFFER_SIZE; j++) {
          curr[tempIdx + j] = buff[j];
        }
        index = 0;
      }
    }
  }

  if (index > 0) {
    EdgeT tempIdx = __sync_fetch_and_add(&currTail, index);
    for (EdgeT j = 0; j < index; j++) {
      curr[tempIdx + j] = buff[j];
    }
  }
#pragma omp barrier
}

// 更新支持边的数值
void UpdateSup(EdgeT e, NodeT *edgesSup, NodeT level, NodeT *buff, EdgeT &index, EdgeT *next, bool *inNext,
               EdgeT &nextTail) {
  NodeT supE = __sync_fetch_and_sub(&edgesSup[e], 1);

  if (supE == (level + 1)) {
    buff[index] = e;
    inNext[e] = true;
    index++;
  }

  if (supE <= level) {
    __sync_fetch_and_add(&edgesSup[e], 1);
  }

  if (index >= BUFFER_SIZE) {
    EdgeT tempIdx = __sync_fetch_and_add(&nextTail, BUFFER_SIZE);
    for (EdgeT bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++) {
      next[tempIdx + bufIdx] = buff[bufIdx];
    }
    index = 0;
  }
}

// 子任务循环迭代消减truss
void SubLevel(const EdgeT *nodeIndex, const NodeT *adj, const EdgeT *curr, bool *inCurr, EdgeT currTail,
              NodeT *edgesSup, NodeT level, EdgeT *next, bool *inNext, EdgeT &nextTail, bool *processed,
              const EdgeT *edgesId, const uint64_t *halfEdges) {
  NodeT buff[BUFFER_SIZE];
  EdgeT index = 0;

#pragma omp for schedule(dynamic, 8)
  for (EdgeT i = 0; i < currTail; i++) {
    EdgeT e1 = curr[i];
    NodeT u = FIRST(halfEdges[e1]);
    NodeT v = SECOND(halfEdges[e1]);

    EdgeT uStart = nodeIndex[u], uEnd = nodeIndex[u + 1];
    EdgeT vStart = nodeIndex[v], vEnd = nodeIndex[v + 1];
    while (uStart < uEnd && vStart < vEnd) {
      if (adj[uStart] < adj[vStart]) {
        ++uStart;
      } else if (adj[uStart] > adj[vStart]) {
        ++vStart;
      } else {
        EdgeT e2 = edgesId[uStart];
        EdgeT e3 = edgesId[vStart];
        ++uStart;
        ++vStart;
        if (processed[e2] || processed[e3]) {
          continue;
        }
        if (edgesSup[e2] > level && edgesSup[e3] > level) {
          UpdateSup(e2, edgesSup, level, buff, index, next, inNext, nextTail);
          UpdateSup(e3, edgesSup, level, buff, index, next, inNext, nextTail);
        } else if (edgesSup[e2] > level) {
          if ((e1 < e3 && inCurr[e3]) || !inCurr[e3]) {
            UpdateSup(e2, edgesSup, level, buff, index, next, inNext, nextTail);
          }
        } else if (edgesSup[e3] > level) {
          if ((e1 < e2 && inCurr[e2]) || !inCurr[e2]) {
            UpdateSup(e3, edgesSup, level, buff, index, next, inNext, nextTail);
          }
        }
      }
    }
  }

  if (index > 0) {
    EdgeT tempIdx = __sync_fetch_and_add(&nextTail, index);
    for (EdgeT bufIdx = 0; bufIdx < index; bufIdx++) {
      next[tempIdx + bufIdx] = buff[bufIdx];
    }
  }
#pragma omp barrier

#pragma omp for
  for (EdgeT i = 0; i < currTail; i++) {
    EdgeT e = curr[i];
    processed[e] = true;
    inCurr[e] = false;
  }
}

// 求解k-truss的主流程
void KTruss(const EdgeT *nodeIndex, const NodeT *adj, const EdgeT *edgesId, const uint64_t *halfEdges,
            EdgeT halfEdgesNum, NodeT *edgesSup, NodeT startLevel) {
  EdgeT currTail = 0;
  EdgeT nextTail = 0;
  NodeT minLevel = halfEdgesNum;
  auto *processed = (bool *)MyMalloc(halfEdgesNum * sizeof(bool));
  auto *curr = (EdgeT *)MyMalloc(halfEdgesNum * sizeof(EdgeT));
  auto *inCurr = (bool *)MyMalloc(halfEdgesNum * sizeof(bool));
  auto *next = (EdgeT *)MyMalloc(halfEdgesNum * sizeof(EdgeT));
  auto *inNext = (bool *)MyMalloc(halfEdgesNum * sizeof(bool));

#pragma omp parallel
  {
    NodeT level = startLevel;
    EdgeT todo = halfEdgesNum;
    if (level > 0u) {
      --level;
      ScanLessThanLevel(halfEdgesNum, edgesSup, level, curr, currTail, inCurr);
#pragma omp single
      { log_debug("level: %u currTail: %u restEdges: %u", level, currTail, todo); }
      while (currTail > 0) {
        todo = todo - currTail;
        SubLevel(nodeIndex, adj, curr, inCurr, currTail, edgesSup, level, next, inNext, nextTail, processed, edgesId,
                 halfEdges);
#pragma omp single
        {
          std::swap(curr, next);
          std::swap(inCurr, inNext);

          currTail = nextTail;
          nextTail = 0;

          log_debug("level: %u currTail: %u restEdges: %u", level, currTail, todo);
        }
#pragma omp barrier
      }
      ++level;
    } else {
#pragma omp for reduction(min : minLevel)
      for (EdgeT i = 0; i < halfEdgesNum; i++) {
        if (edgesSup[i] < minLevel) {
          minLevel = edgesSup[i];
        }
      }
      level = minLevel;
    }
    while (todo > 0) {
      Scan(halfEdgesNum, edgesSup, level, curr, currTail, inCurr);
#pragma omp single
      { log_debug("level: %u currTail: %u restEdges: %u", level, currTail, todo); }
#pragma omp barrier

      while (currTail > 0) {
        todo = todo - currTail;
        SubLevel(nodeIndex, adj, curr, inCurr, currTail, edgesSup, level, next, inNext, nextTail, processed, edgesId,
                 halfEdges);
#pragma omp single
        {
          std::swap(curr, next);
          std::swap(inCurr, inNext);

          currTail = nextTail;
          nextTail = 0;

          log_debug("level: %u currTail: %u restEdges: %u", level, currTail, todo);
        }
#pragma omp barrier
      }
      ++level;
#pragma omp barrier
    }
  }

  MyFree((void *&)processed, halfEdgesNum * sizeof(bool));
  MyFree((void *&)curr, halfEdgesNum * sizeof(EdgeT));
  MyFree((void *&)inCurr, halfEdgesNum * sizeof(bool));
  MyFree((void *&)next, halfEdgesNum * sizeof(EdgeT));
  MyFree((void *&)inNext, halfEdgesNum * sizeof(bool));
}

// 获取各层次truss的边的数量
NodeT DisplayStats(const NodeT *edgesSup, EdgeT halfEdgesNum, NodeT minK) {
  NodeT maxSup = 0;

#pragma omp parallel for reduction(max : maxSup)
  for (EdgeT i = 0; i < halfEdgesNum; i++) {
    if (maxSup < edgesSup[i]) {
      maxSup = edgesSup[i];
    }
  }

  EdgeT numEdgesWithMaxSup = 0;
#pragma omp parallel for reduction(+ : numEdgesWithMaxSup)
  for (EdgeT i = 0; i < halfEdgesNum; i++) {
    if (edgesSup[i] == maxSup) {
      numEdgesWithMaxSup++;
    }
  }

  log_info("Max-truss: %u  Edges in Max-truss: %u", maxSup + 2, numEdgesWithMaxSup);
  if (maxSup + 2 >= minK) {
    printf("kmax = %u, Edges in kmax-truss = %u.\n", maxSup + 2, numEdgesWithMaxSup);
  }
  return maxSup + 2;
}
