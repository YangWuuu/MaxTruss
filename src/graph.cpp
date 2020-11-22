#include "graph.h"

#include <algorithm>
#include <cstring>
#include <map>

#pragma ide diagnostic ignored "openmp-use-default-none"

NodeT Graph::GetMaxCore() {
  log_info(coreClock_.Start());

  ::CalDeg(rawEdges_, rawEdgesNum_, rawNodesNum_, rawDeg_);
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

  log_info(coreClock_.Count("maxK: %u", maxCoreNum));
  return maxCoreNum;
}

// 获取max-k-truss主流程
NodeT Graph::KMaxTruss(NodeT startK, NodeT startLevel) {
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
           edgesSup_, startLevel);
  log_info(trussClock_.Count("KTruss"));

  // 打印信息
  NodeT possibleKMax = DisplayStats(edgesSup_, halfEdgesNum_, startK_);

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

    ::CalDeg(edges_, edgesNum_, nodesNum_, deg_);
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

  ::Unzip(halfEdges_, halfEdgesNum_, halfEdgesFirst_, halfEdgesSecond_);
  ::CalDeg(halfEdges_, halfEdgesNum_, nodesNum_, halfDeg_);
  ::NodeIndex(halfDeg_, nodesNum_, halfNodeIndex_);
  log_info(preprocessClock_.Count("half"));

  ::GetEdgesId(edges_, edgesNum_, edgesId_, halfNodeIndex_, halfEdgesSecond_);
  log_info(preprocessClock_.Count("GetEdgesId"));
}

// 图的裁剪
void Graph::RemoveEdges() {
  edges_ = (uint64_t *)malloc(rawEdgesNum_ * sizeof(uint64_t));
  // TODO parallel
  edgesNum_ = std::copy_if(rawEdges_, rawEdges_ + rawEdgesNum_, edges_,
                           [&](const uint64_t edge) {
                             return rawCore_[FIRST(edge)] >= (startK_ - 2) &&
                                    rawCore_[SECOND(edge)] >= (startK_ - 2);
                           }) -
              edges_;

  log_info(preprocessClock_.Count("edgesNum_: %u", edgesNum_));
}

// 三角形计数
void Graph::TriCount() {
  log_info(triCountClock_.Start());

  GetEdgeSup(halfEdgesNum_, halfEdgesFirst_, halfEdgesSecond_, halfDeg_,
             halfNodeIndex_, edgesSup_);
  log_info(triCountClock_.Count("Count"));

  // TODO can remove
  uint64_t count = 0;
  for (uint64_t i = 0; i < halfEdgesNum_; i++) {
    count += edgesSup_[i];
  }
  log_info(triCountClock_.Count("triangle count: %lu", count / 3));
}
