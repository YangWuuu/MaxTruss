#include "graph.h"

#include <algorithm>
#include <map>

#pragma ide diagnostic ignored "openmp-use-default-none"

NodeT Graph::GetMaxK() {
  log_info(preprocessClock.Start());
  rawDeg_ = (NodeT *)calloc(rawNodesNum_, sizeof(NodeT));
  ::CalDeg(rawEdges_, rawEdgesNum_, rawDeg_);
  log_info(preprocessClock.Count("rawDeg_"));

//  auto *degNum = (EdgeT *)calloc(rawNodesNum_, sizeof(EdgeT));
//  //here sync fetch add is slow
//  //#ifndef SERIAL
//  //#pragma omp parallel for
//  //#endif
//  for (int i = 0; i < rawNodesNum_; i++) {
//    //    __sync_fetch_and_add(&degNum[rawDeg_[i]], 1);
//    degNum[rawDeg_[i]]++;
//  }
//  log_info(preprocessClock.Count("degNum"));
//
//  NodeT maxK = 0;
//  NodeT reverseCount = 0;
//  for (auto m = rawNodesNum_ - 1; m != 0; m--) {
//    NodeT proposedKMax = m + 1;
//    reverseCount += degNum[m];
//    if (reverseCount >= proposedKMax) {
//      maxK = proposedKMax;
//      break;
//    }
//  }

  // TODO parallel
  std::map<NodeT, NodeT> degNum;
  for (int i = 0; i < rawNodesNum_; i++) {
    if (degNum.count(rawDeg_[i]) == 0) {
      degNum[rawDeg_[i]] = 0;
    }
    degNum[rawDeg_[i]]++;
  }
  log_info(preprocessClock.Count("degNum"));

  NodeT maxK = 0;
  NodeT reverseCount = 0;
  for (auto m = degNum.rbegin(); m != degNum.rend(); m++) {
    NodeT proposedKMax = m->first + 1;
    reverseCount += m->second;
    if (reverseCount >= proposedKMax) {
      maxK = proposedKMax;
      break;
    }
  }

  log_info(preprocessClock.Count("maxK: %u", maxK));
  return maxK;
}

// 获取max-k-truss主流程
bool Graph::MaxKTruss(NodeT startK) {
  startK_ = startK;

  // 预处理
  Preprocess();

  // 三角形计数
  TriCount();

  // 求解k-truss
  log_info(trussClock.Start());
  ::KTruss(nodeIndex_, edgesSecond_, edgesId_, halfEdges_, halfEdgesNum_,
           edgesSup_);
  log_info(trussClock.Count("KTruss"));

  // 打印信息
  bool isValid = displayStats(edgesSup_, halfEdgesNum_, startK_);
  log_info(trussClock.Count("displayStats isValid: %d", isValid));

  return isValid;
}

// 图的预处理
void Graph::Preprocess() {
  log_info(preprocessClock.Start());
  log_info(preprocessClock.Count("startK_: %u", startK_));

  if (startK_ > 10u) {
    RemoveEdges();
    nodesNum_ = FIRST(edges_[edgesNum_ - 1]) + 1;
    log_info(preprocessClock.Count("nodesNum_: %u", nodesNum_));
    deg_ = (NodeT *)calloc(nodesNum_, sizeof(NodeT));
    ::CalDeg(edges_, edgesNum_, deg_);
    log_info(preprocessClock.Count("cal deg"));
  } else {
    startK_ = 0;
    edges_ = rawEdges_;
    edgesNum_ = rawEdgesNum_;
    nodesNum_ = rawNodesNum_;
    deg_ = rawDeg_;
    log_info(preprocessClock.Count("nodesNum_: %u", nodesNum_));
  }

  ::Unzip(edges_, edgesNum_, edgesFirst_, edgesSecond_);
  log_info(preprocessClock.Count("Unzip"));

  halfEdgesNum_ = edgesNum_ / 2;
  halfEdges_ = (uint64_t *)malloc(halfEdgesNum_ * sizeof(uint64_t));
  // TODO parallel
  std::copy_if(edges_, edges_ + edgesNum_, halfEdges_,
               [](const uint64_t &edge) { return FIRST(edge) < SECOND(edge); });
  log_info(preprocessClock.Count("halfEdgesNum_: %u", halfEdgesNum_));

  ::NodeIndex(deg_, nodesNum_, nodeIndex_);
  log_info(preprocessClock.Count("NodeIndex"));

  GetEdgesId();
  log_info(preprocessClock.Count("GetEdgesId"));
}

// 图的裁剪
void Graph::RemoveEdges() {
    edges_ = (uint64_t *)malloc(rawEdgesNum_ * sizeof(uint64_t));
  // TODO parallel
  // TODO 这里可以做的更彻底一点
  edgesNum_ = std::copy_if(rawEdges_, rawEdges_ + rawEdgesNum_, edges_,
                           [&](const uint64_t edge) {
                             return rawDeg_[FIRST(edge)] > startK_ &&
                                    rawDeg_[SECOND(edge)] > startK_;
                           }) -
              edges_;

  log_info(preprocessClock.Count("edgesNum_: %u", edgesNum_));
}

// 三角形计数
void Graph::TriCount() {
  log_info(triCountClock.Start());
  edgesSup_ = (EdgeT *)calloc(halfEdgesNum_, sizeof(EdgeT));
  GetEdgeSup(halfEdges_, halfEdgesNum_, halfEdgesFirst_, halfEdgesSecond_,
             halfDeg_, nodesNum_, halfNodeIndex_, edgesSup_);
  log_info(triCountClock.Count("Count"));

  // TODO can remove
  uint64_t count = 0;
  for (uint64_t i = 0; i < halfEdgesNum_; i++) {
    count += edgesSup_[i];
  }
  log_info(triCountClock.Count("triangle count: %lu", count / 3));
}

// 边编号
void Graph::GetEdgesId() {
  edgesId_ = (EdgeT *)malloc(edgesNum_ * sizeof(EdgeT));

#ifdef SERIAL
  auto *nodeIndexCopy = (EdgeT *)malloc((nodesNum_ + 1) * sizeof(EdgeT));
  for (NodeT i = 0; i < nodesNum_ + 1; i++) {
    nodeIndexCopy[i] = nodeIndex_[i];
  }
  EdgeT edgeId = 0;
  for (NodeT u = 0; u < nodesNum_; u++) {
    for (EdgeT j = nodeIndex_[u]; j < nodeIndex_[u + 1]; j++) {
      NodeT v = edgesSecond_[j];
      if (u < v) {
        edgesId_[j] = edgeId;
        nodeIndexCopy[u]++;
        if (edgesSecond_[nodeIndexCopy[v]] == u) {
          edgesId_[nodeIndexCopy[v]] = edgeId;
          nodeIndexCopy[v]++;
        }
        edgeId++;
      }
    }
  }
#else
  auto *nodeIndexCopy = (EdgeT *)malloc((nodesNum_ + 1) * sizeof(EdgeT));
  nodeIndexCopy[0] = 0;
  auto *upper_tri_start = (EdgeT *)malloc(nodesNum_ * sizeof(EdgeT));

  auto num_threads = omp_get_max_threads();
  std::vector<uint32_t> histogram(CACHE_LINE_ENTRY * num_threads);

#pragma omp parallel
  {
#pragma omp for
    // Histogram (Count).
    for (NodeT u = 0; u < nodesNum_; u++) {
      upper_tri_start[u] =
          (nodeIndex_[u + 1] - nodeIndex_[u] > 256)
              ? GallopingSearch(edgesSecond_, nodeIndex_[u], nodeIndex_[u + 1],
                                u)
              : LinearSearch(edgesSecond_, nodeIndex_[u], nodeIndex_[u + 1], u);
    }

    // Scan.
    InclusivePrefixSumOMP(histogram, nodeIndexCopy + 1, nodesNum_, [&](int u) {
      return nodeIndex_[u + 1] - upper_tri_start[u];
    });

    // Transform.
    NodeT u = 0;
#pragma omp for schedule(dynamic, 6000)
    for (EdgeT j = 0u; j < edgesNum_; j++) {
      u = edgesFirst_[j];
      if (j < upper_tri_start[u]) {
        auto v = edgesSecond_[j];
        auto offset = BranchFreeBinarySearch(edgesSecond_, nodeIndex_[v],
                                             nodeIndex_[v + 1], u);
        auto eid = nodeIndexCopy[v] + (offset - upper_tri_start[v]);
        edgesId_[j] = eid;
      } else {
        edgesId_[j] = nodeIndexCopy[u] + (j - upper_tri_start[u]);
      }
    }
  }
  free(upper_tri_start);
  free(nodeIndexCopy);
#endif
}

// 计算节点的度
void CalDeg(const uint64_t *edges, EdgeT edgesNum, NodeT *deg) {
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
  edgesFirst = (NodeT *)malloc(edgesNum * sizeof(NodeT));
  edgesSecond = (NodeT *)malloc(edgesNum * sizeof(NodeT));
#ifndef SERIAL
#pragma omp parallel for
#endif
  for (EdgeT i = 0; i < edgesNum; i++) {
    edgesFirst[i] = FIRST(edges[i]);
    edgesSecond[i] = SECOND(edges[i]);
  }
}

// 转换CSR格式
void NodeIndex(const NodeT *deg, NodeT nodesNum, EdgeT *&nodeIndex) {
  nodeIndex = (EdgeT *)calloc((nodesNum + 1), sizeof(EdgeT));
  // TODO parallel
  for (NodeT i = 0; i < nodesNum; i++) {
    nodeIndex[i + 1] = nodeIndex[i] + deg[i];
  }
}

// 三角形计数获取支持边数量
void GetEdgeSup(const uint64_t *halfEdges, EdgeT halfEdgesNum,
                NodeT *&halfEdgesFirst, NodeT *&halfEdgesSecond,
                NodeT *&halfDeg, NodeT nodesNum, EdgeT *&halfNodeIndex,
                NodeT *edgesSup) {
  ::Unzip(halfEdges, halfEdgesNum, halfEdgesFirst, halfEdgesSecond);
  halfDeg = (NodeT *)calloc(nodesNum, sizeof(NodeT));
  ::CalDeg(halfEdges, halfEdgesNum, halfDeg);
  ::NodeIndex(halfDeg, nodesNum, halfNodeIndex);

#ifndef SERIAL
#pragma omp parallel for schedule(dynamic, 1024)
#endif
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
bool displayStats(const EdgeT *EdgeSupport, EdgeT halfEdgesNum, NodeT minK) {
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
  if (numEdgesWithMaxSup == 0 || maxSup < minK) {
    return false;
  }
  printf("kmax = %u, Edges in kmax-truss = %u.\n", maxSup + 2,
         numEdgesWithMaxSup);
  return true;
}
