#pragma once

#include "clock.h"
#include "log.h"
#include "util.h"

class Graph {
 public:
  Graph(uint64_t *edges, EdgeT edgesNum);

  ~Graph();

  // 获取最大度
  NodeT GetMaxCore();

  // 获取max-k-truss
  NodeT KMaxTruss(NodeT startK, NodeT startLevel);

 private:
  // 图的预处理
  void Preprocess(NodeT startK);

  void FreeRawGraph();
  void FreeGraph();
  void FreeHalfGraph();

  // 计时
  Clock coreClock_;
  Clock preprocessClock_;
  Clock triCountClock_;
  Clock trussClock_;

  // 原始图信息
  uint64_t *rawEdges_{nullptr};
  EdgeT rawEdgesNum_{};
  NodeT rawNodesNum_{};
  NodeT *rawCore_{nullptr};
  EdgeT *rawNodeIndex_{nullptr};
  NodeT *rawAdj_{nullptr};
  uint64_t *cudaRawEdges_{nullptr};

  // 新图信息
  uint64_t *edges_{nullptr};
  EdgeT edgesNum_{};
  NodeT nodesNum_{};
  EdgeT *nodeIndex_{nullptr};
  NodeT *adj_{nullptr};

  // 有向图信息
  uint64_t *halfEdges_{nullptr};
  EdgeT halfEdgesNum_{};
  NodeT halfNodesNum_{};
  EdgeT *halfNodeIndex_{nullptr};
  NodeT *halfAdj_{nullptr};

  // 边编号
  EdgeT *edgesId_{nullptr};
  // 支持边
  NodeT *edgesSup_{nullptr};
};

// 计算KCore
NodeT KCore(const EdgeT *nodeIndex, const NodeT *adj, NodeT nodesNum, NodeT *&core);

// 图的裁剪
EdgeT ConstructNewGraph(const uint64_t *rawEdges, uint64_t *&edges, const NodeT *rawCore, EdgeT rawEdgesNum,
                        NodeT startK);

// 构造CSR
void ConstructCSRGraph(const uint64_t *edges, EdgeT edgesNum, EdgeT *&nodeIndex, NodeT *&adj);

// 构造有向图
void ConstructHalfEdges(const uint64_t *edges, uint64_t *&halfEdges, EdgeT halfEdgesNum);

// 获取边序号
void GetEdgesId(const uint64_t *edges, EdgeT edgesNum, const EdgeT *halfNodeIndex, const NodeT *halfAdj,
                EdgeT *&edgesId);

// 三角形计数获取支持边数量
void GetEdgeSup(const EdgeT *halfNodeIndex, const NodeT *halfAdj, NodeT halfNodesNum, NodeT *&edgesSup);
void GetEdgeSup(const EdgeT *nodeIndex, const NodeT *adj, const EdgeT *edgesId, NodeT nodesNum, NodeT *&edgesSup);

// 求解k-truss的主流程
void KTruss(const EdgeT *nodeIndex, const NodeT *adj, const EdgeT *edgesId, const uint64_t *halfEdges,
            EdgeT halfEdgesNum, NodeT *edgesSup, NodeT startLevel);
void KTruss(EdgeT *nodeIndex, NodeT *adj, EdgeT *edgesId, NodeT nodesNum, const uint64_t *halfEdges, EdgeT halfEdgesNum,
            NodeT *edgesSup, NodeT startLevel);

// 获取各层次truss的边的数量
NodeT DisplayStats(const NodeT *edgesSup, EdgeT halfEdgesNum, NodeT minK);
