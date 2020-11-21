#include "graph.h"

#include <algorithm>
#include <cstring>
#include <map>

#pragma ide diagnostic ignored "openmp-use-default-none"

NodeT Graph::GetMaxK() {
  log_info(coreClock_.Start());

  rawDeg_ = (NodeT *)calloc(rawNodesNum_, sizeof(NodeT));
  ::CalDeg(rawEdges_, rawEdgesNum_, rawDeg_);
  log_info(coreClock_.Count("rawDeg_"));

  ::Unzip(rawEdges_, rawEdgesNum_, rawEdgesFirst_, rawEdgesSecond_);
  log_info(coreClock_.Count("Unzip"));

  ::NodeIndex(rawDeg_, rawNodesNum_, rawNodeIndex_);
  log_info(coreClock_.Count("NodeIndex"));

  rawCore_ = (NodeT *)malloc(rawNodesNum_ * sizeof(NodeT));
  // TODO parallel
  memcpy(rawCore_, rawDeg_, rawNodesNum_ * sizeof(NodeT));
  log_info(coreClock_.Count("rawCore_"));

  KCore(rawNodeIndex_, rawEdgesSecond_, rawNodesNum_, rawCore_);
  log_info(coreClock_.Count("KCore"));

  NodeT maxCoreNum = 0;
#pragma omp parallel for reduction(max : maxCoreNum)
  for (NodeT i = 0; i < rawNodesNum_; i++) {
    maxCoreNum = std::max(maxCoreNum, rawCore_[i]);
  }
  ++maxCoreNum;
  auto *histogram = (NodeT *)calloc(maxCoreNum, sizeof(NodeT));
#pragma omp parallel for
  for (NodeT i = 0; i < rawNodesNum_; i++) {
    NodeT coreVal = rawCore_[i];
    __sync_fetch_and_add(&histogram[coreVal], 1);
  }
  log_info(coreClock_.Count("histogram"));

  log_info(coreClock_.Count("maxK: %u", maxCoreNum - 1));
  return maxCoreNum - 1;
}

// 获取max-k-truss主流程
NodeT Graph::MaxKTruss(NodeT startK) {
  startK_ = startK;

  // 预处理
  Preprocess();
  if (edgesNum_ == 0) {
    return 0;
  }

  // 三角形计数
  TriCount();

  // 求解k-truss
  log_info(trussClock_.Start());
  ::KTruss(nodeIndex_, edgesSecond_, edgesId_, halfEdges_, halfEdgesNum_,
           edgesSup_);
  log_info(trussClock_.Count("KTruss"));

  // 打印信息
  NodeT possibleKMax = displayStats(edgesSup_, halfEdgesNum_, startK_);

  return possibleKMax;
}

// 图的预处理
void Graph::Preprocess() {
  log_info(preprocessClock_.Start());
  log_info(preprocessClock_.Count("startK_: %u", startK_));

  if (startK_ > 0u) {
    RemoveEdges();
    if (edgesNum_ == 0) {
      return;
    }
    nodesNum_ = FIRST(edges_[edgesNum_ - 1]) + 1;
    log_info(preprocessClock_.Count("nodesNum_: %u", nodesNum_));

    deg_ = (NodeT *)calloc(nodesNum_, sizeof(NodeT));
    ::CalDeg(edges_, edgesNum_, deg_);
    log_info(preprocessClock_.Count("cal deg"));

    ::Unzip(edges_, edgesNum_, edgesFirst_, edgesSecond_);
    log_info(preprocessClock_.Count("Unzip"));

    ::NodeIndex(deg_, nodesNum_, nodeIndex_);
    log_info(preprocessClock_.Count("NodeIndex"));
  } else {
    startK_ = 0;
    edges_ = rawEdges_;
    edgesNum_ = rawEdgesNum_;
    nodesNum_ = rawNodesNum_;
    deg_ = rawDeg_;
    edgesFirst_ = rawEdgesFirst_;
    edgesSecond_ = rawEdgesSecond_;
    nodeIndex_ = rawNodeIndex_;
    log_info(preprocessClock_.Count("nodesNum_: %u", nodesNum_));
  }

  halfEdgesNum_ = edgesNum_ / 2;
  halfEdges_ = (uint64_t *)malloc(halfEdgesNum_ * sizeof(uint64_t));
  // TODO parallel
  std::copy_if(edges_, edges_ + edgesNum_, halfEdges_,
               [](const uint64_t &edge) { return FIRST(edge) < SECOND(edge); });
  log_info(preprocessClock_.Count("halfEdgesNum_: %u", halfEdgesNum_));

  GetEdgesId();
  log_info(preprocessClock_.Count("GetEdgesId"));
}

// 图的裁剪
void Graph::RemoveEdges() {
  edges_ = (uint64_t *)malloc(rawEdgesNum_ * sizeof(uint64_t));
  // TODO parallel
  edgesNum_ = std::copy_if(rawEdges_, rawEdges_ + rawEdgesNum_, edges_,
                           [&](const uint64_t edge) {
                             return rawCore_[FIRST(edge)] > startK_ &&
                                    rawCore_[SECOND(edge)] > startK_;
                           }) -
              edges_;

  log_info(preprocessClock_.Count("edgesNum_: %u", edgesNum_));
}

// 三角形计数
void Graph::TriCount() {
  log_info(triCountClock_.Start());
  edgesSup_ = (EdgeT *)calloc(halfEdgesNum_, sizeof(EdgeT));
  GetEdgeSup(halfEdges_, halfEdgesNum_, halfEdgesFirst_, halfEdgesSecond_,
             halfDeg_, nodesNum_, halfNodeIndex_, edgesSup_);
  log_info(triCountClock_.Count("Count"));

  // TODO can remove
  uint64_t count = 0;
  for (uint64_t i = 0; i < halfEdgesNum_; i++) {
    count += edgesSup_[i];
  }
  log_info(triCountClock_.Count("triangle count: %lu", count / 3));
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
        edgesId_[nodeIndexCopy[v]] = edgeId;
        nodeIndexCopy[v]++;
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
