#pragma once

#include "clock.h"
#include "log.h"
#include "util.h"

class Graph {
 public:
  Graph(uint64_t *edges, EdgeT edgesNum)
      : rawEdges_(edges),
        rawEdgesNum_(edgesNum),
        preprocessClock("Preprocess"),
        triCountClock("TriCount"),
        trussClock("Truss") {
    rawNodesNum_ = FIRST(rawEdges_[rawEdgesNum_ - 1]) + 1;
  }

  ~Graph() {}

  bool MaxKTruss(bool remove = false);

 private:
  void Preprocess(bool remove);
  void RemoveEdges();
  void GetEdgesId();

  void TriCount();

  Clock preprocessClock;
  Clock triCountClock;
  Clock trussClock;

  bool repeat_{false};
  NodeT minK_{2};

  uint64_t *rawEdges_;
  EdgeT rawEdgesNum_;
  NodeT rawNodesNum_;
  NodeT *rawDeg_{nullptr};

  uint64_t *edges_{nullptr};
  EdgeT edgesNum_{};
  NodeT nodesNum_{};
  NodeT *deg_{nullptr};
  EdgeT *nodeIndex_{nullptr};

  NodeT *edgesFirst_{nullptr};
  NodeT *edgesSecond_{nullptr};

  uint64_t *halfEdges_{nullptr};
  EdgeT halfEdgesNum_{};

  EdgeT *edgesId_{nullptr};
  EdgeT *edgesSup_{nullptr};
};

void CalDeg(const uint64_t *edges, EdgeT edgesNum, NodeT *deg);
void Unzip(const uint64_t *edges, EdgeT edgesNum, NodeT *&edgesFirst,
           NodeT *&edgesSecond);
void NodeIndex(const NodeT *deg, NodeT nodesNum, EdgeT *&nodeIndex);

void GetEdgeSup(const EdgeT *nodeIndex, const NodeT *edgesSecond,
                const EdgeT *edgesId, NodeT nodesNum, NodeT *edgesSup);

void KTruss(const EdgeT *nodeIndex, const NodeT *edgesSecond,
            const EdgeT *edgesId, const uint64_t *halfEdges, EdgeT halfEdgesNum,
            EdgeT *edgesSup);

bool displayStats(const EdgeT *EdgeSupport, EdgeT halfEdgesNum, NodeT minK);