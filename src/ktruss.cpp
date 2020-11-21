#include "util.h"

#pragma ide diagnostic ignored "openmp-use-default-none"

// 并行扫描支持边是否与truss层次相同
void Scan(EdgeT numEdges, const EdgeT *edgesSup, int level, EdgeT *curr,
          EdgeT &currTail, bool *inCurr) {
  NodeT buff[BUFFER_SIZE];
  EdgeT index = 0;

#pragma omp for
  for (EdgeT i = 0; i < numEdges; i++) {
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

// 更新支持边的数值
void UpdateSup(EdgeT e, EdgeT *edgesSup, int level, NodeT *buff, EdgeT &index,
               EdgeT *next, bool *inNext, EdgeT &nextTail) {
  int supE = __sync_fetch_and_sub(&edgesSup[e], 1);

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
    for (EdgeT bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++)
      next[tempIdx + bufIdx] = buff[bufIdx];
    index = 0;
  }
}

// 子任务循环迭代消减truss
void SubLevel(const EdgeT *nodeIndex, const NodeT *edgesSecond,
              const EdgeT *curr, bool *inCurr, EdgeT currTail, EdgeT *edgesSup,
              int level, EdgeT *next, bool *inNext, EdgeT &nextTail,
              bool *processed, const EdgeT *edgesId,
              const uint64_t *halfEdges) {
  NodeT buff[BUFFER_SIZE];
  EdgeT index = 0;

#pragma omp for schedule(dynamic, 8)
  for (EdgeT i = 0; i < currTail; i++) {
    // process edge <u,v>
    EdgeT e1 = curr[i];
    NodeT u = FIRST(halfEdges[e1]);
    NodeT v = SECOND(halfEdges[e1]);

    EdgeT uStart = nodeIndex[u], uEnd = nodeIndex[u + 1];
    EdgeT vStart = nodeIndex[v], vEnd = nodeIndex[v + 1];
    while (uStart < uEnd && vStart < vEnd) {
      if (edgesSecond[uStart] < edgesSecond[vStart]) {
        ++uStart;
      } else if (edgesSecond[uStart] > edgesSecond[vStart]) {
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
void KTruss(const EdgeT *nodeIndex, const NodeT *edgesSecond,
            const EdgeT *edgesId, const uint64_t *halfEdges, EdgeT halfEdgesNum,
            EdgeT *edgesSup) {
  EdgeT currTail = 0;
  EdgeT nextTail = 0;
  auto *processed = (bool *)calloc(halfEdgesNum, sizeof(bool));
  auto *curr = (EdgeT *)calloc(halfEdgesNum, sizeof(EdgeT));
  auto *inCurr = (bool *)calloc(halfEdgesNum, sizeof(bool));
  auto *next = (EdgeT *)calloc(halfEdgesNum, sizeof(EdgeT));
  auto *inNext = (bool *)calloc(halfEdgesNum, sizeof(bool));

#ifndef SERIAL
#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();

    int level = 0;
    EdgeT todo = halfEdgesNum;
    while (todo > 0) {
      Scan(halfEdgesNum, edgesSup, level, curr, currTail, inCurr);
      while (currTail > 0) {
        todo = todo - currTail;
        SubLevel(nodeIndex, edgesSecond, curr, inCurr, currTail, edgesSup,
                 level, next, inNext, nextTail, processed, edgesId, halfEdges);
        if (tid == 0) {
          EdgeT *tempCurr = curr;
          curr = next;
          next = tempCurr;

          bool *tempInCurr = inCurr;
          inCurr = inNext;
          inNext = tempInCurr;

          currTail = nextTail;
          nextTail = 0;

          log_debug("level: %d restEdges: %lu", level, todo);
        }
#pragma omp barrier
      }
      level = level + 1;
#pragma omp barrier
    }
  }
}

// 获取各层次truss的边的数量
NodeT displayStats(const EdgeT *EdgeSupport, EdgeT halfEdgesNum, NodeT minK) {
  NodeT minSup = std::numeric_limits<NodeT>::max();
  NodeT maxSup = 0;

  for (EdgeT i = 0; i < halfEdgesNum; i++) {
    if (minSup > EdgeSupport[i]) {
      minSup = EdgeSupport[i];
    }
    if (maxSup < EdgeSupport[i]) {
      maxSup = EdgeSupport[i];
    }
  }

  EdgeT numEdgesWithMinSup = 0;
  EdgeT numEdgesWithMaxSup = 0;
  for (EdgeT i = 0; i < halfEdgesNum; i++) {
    if (EdgeSupport[i] == minSup) {
      numEdgesWithMinSup++;
    }
    if (EdgeSupport[i] == maxSup) {
      numEdgesWithMaxSup++;
    }
  }

  std::vector<uint64_t> sups(maxSup + 1);
  for (EdgeT i = 0; i < halfEdgesNum; i++) {
    sups[EdgeSupport[i]]++;
  }
  for (int i = 0; i < maxSup + 1; i++) {
    if (sups[i] > 0) {
      log_debug("k: %d  edges: %lu", i + 2, sups[i]);
    }
  }

  log_info("Min-truss: %u  Edges in Min-truss: %u", minSup + 2,
           numEdgesWithMinSup);
  log_info("Max-truss: %u  Edges in Max-truss: %u", maxSup + 2,
           numEdgesWithMaxSup);
  if (maxSup + 2 >= minK) {
    printf("kmax = %u, Edges in kmax-truss = %u.\n", maxSup + 2,
           numEdgesWithMaxSup);
  }
  return maxSup + 2;
}
