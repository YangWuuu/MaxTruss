#pragma once

#include "clock.h"
#include "log.h"
#include "util.h"

class Graph {
 public:
  Graph(uint64_t *edges, EdgeT edgesNum)
      : rawEdges_(edges),
        rawEdgesNum_(edgesNum),
        coreClock_("kCore"),
        preprocessClock_("Preprocess"),
        triCountClock_("TriCount"),
        trussClock_("Truss") {
    rawNodesNum_ = FIRST(rawEdges_[rawEdgesNum_ - 1]) + 1;
  }

  ~Graph() {}

  NodeT GetMaxK();

  // 获取max-k-truss
  NodeT MaxKTruss(NodeT startK);

 private:
  // 图的预处理
  void Preprocess();
  // 图的裁剪
  void RemoveEdges();
  // 边编号
  void GetEdgesId();
  // 三角形计数
  void TriCount();

  // 计时
  Clock coreClock_;
  Clock preprocessClock_;
  Clock triCountClock_;
  Clock trussClock_;

  NodeT startK_{0};

  // 原始图信息
  uint64_t *rawEdges_;
  EdgeT rawEdgesNum_;
  NodeT rawNodesNum_;
  NodeT *rawDeg_{nullptr};
  NodeT *rawCore_{nullptr};
  NodeT *rawEdgesFirst_{nullptr};
  NodeT *rawEdgesSecond_{nullptr};
  EdgeT *rawNodeIndex_{nullptr};

  // 新图信息
  uint64_t *edges_{nullptr};
  EdgeT edgesNum_{};
  NodeT nodesNum_{};
  NodeT *edgesFirst_{nullptr};
  NodeT *edgesSecond_{nullptr};
  NodeT *deg_{nullptr};
  EdgeT *nodeIndex_{nullptr};

  // 有向图信息
  uint64_t *halfEdges_{nullptr};
  EdgeT halfEdgesNum_{};
  NodeT *halfEdgesFirst_{nullptr};
  NodeT *halfEdgesSecond_{nullptr};
  NodeT *halfDeg_{nullptr};
  EdgeT *halfNodeIndex_{nullptr};

  // 边编号
  EdgeT *edgesId_{nullptr};
  // 支持边
  EdgeT *edgesSup_{nullptr};
};

// 计算节点的度
void CalDeg(const uint64_t *edges, EdgeT edgesNum, NodeT *deg);

// 边的解压缩
void Unzip(const uint64_t *edges, EdgeT edgesNum, NodeT *&edgesFirst,
           NodeT *&edgesSecond);

// 转换CSR格式
void NodeIndex(const NodeT *deg, NodeT nodesNum, EdgeT *&nodeIndex);

// 三角形计数获取支持边数量
void GetEdgeSup(const uint64_t *halfEdges, EdgeT halfEdgesNum,
                NodeT *&halfEdgesFirst, NodeT *&halfEdgesSecond,
                NodeT *&halfDeg, NodeT nodesNum, EdgeT *&halfNodeIndex,
                NodeT *edgesSup);

// 求解k-truss的主流程
void KTruss(const EdgeT *nodeIndex, const NodeT *edgesSecond,
            const EdgeT *edgesId, const uint64_t *halfEdges, EdgeT halfEdgesNum,
            EdgeT *edgesSup);

// 获取各层次truss的边的数量
NodeT displayStats(const EdgeT *EdgeSupport, EdgeT halfEdgesNum, NodeT minK);

void KCore(const EdgeT *nodeIndex, const NodeT *edgesSecond, NodeT nodesNum,
           NodeT *deg);
