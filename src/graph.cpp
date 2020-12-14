#include "graph.h"

#include <map>

Graph::Graph(uint64_t *edges, EdgeT edgesNum)
    : coreClock_("kCore"),
      preprocessClock_("Preprocess"),
      triCountClock_("TriCount"),
      trussClock_("Truss"),
      rawEdges_(edges),
      rawEdgesNum_(edgesNum) {
  rawNodesNum_ = FIRST(rawEdges_[rawEdgesNum_ - 1]) + 1;
#ifdef CUDA
  log_info("start malloc");
  CUDA_TRY(cudaMallocManaged((void **)&cudaRawEdges_, rawEdgesNum_ * sizeof(uint64_t)));
  log_info("end malloc");
  CUDA_TRY(cudaMemcpy(cudaRawEdges_, rawEdges_, rawEdgesNum_ * sizeof(uint64_t), cudaMemcpyHostToDevice));
  log_info("end memcpy");
#else
  cudaRawEdges_ = rawEdges_;
#endif
}

void Graph::FreeRawGraph() {
#ifdef CUDA
  if (cudaRawEdges_) {
    CUDA_TRY(cudaFree(cudaRawEdges_));
    cudaRawEdges_ = nullptr;
  }
  if (rawCore_) {
    CUDA_TRY(cudaFree(rawCore_));
    rawCore_ = nullptr;
  }
  if (rawNodeIndex_) {
    CUDA_TRY(cudaFree(rawNodeIndex_));
    rawNodeIndex_ = nullptr;
  }
  if (rawAdj_) {
    CUDA_TRY(cudaFree(rawAdj_));
    rawAdj_ = nullptr;
  }
#else
  if (rawEdges_) {
    MyFree((void *&)rawEdges_, rawEdgesNum_ * sizeof(uint64_t));
  }
  if (rawCore_) {
    MyFree((void *&)rawCore_, rawNodesNum_ * sizeof(NodeT));
  }
  if (rawNodeIndex_) {
    MyFree((void *&)rawNodeIndex_, (rawNodesNum_ + 1) * sizeof(EdgeT));
  }
  if (rawAdj_) {
    MyFree((void *&)rawAdj_, rawEdgesNum_ * sizeof(NodeT));
  }
#endif
}

void Graph::FreeGraph() {
#ifdef CUDA
  if (edges_) {
    CUDA_TRY(cudaFree(edges_));
    edges_ = nullptr;
  }
  if (nodeIndex_) {
    CUDA_TRY(cudaFree(nodeIndex_));
    nodeIndex_ = nullptr;
  }
  if (adj_) {
    CUDA_TRY(cudaFree(adj_));
    adj_ = nullptr;
  }
#else
  if (edges_) {
    MyFree((void *&)edges_, edgesNum_ * sizeof(uint64_t));
  }
  if (nodeIndex_) {
    MyFree((void *&)nodeIndex_, (nodesNum_ + 1) * sizeof(EdgeT));
  }
  if (adj_) {
    MyFree((void *&)adj_, edgesNum_ * sizeof(NodeT));
  }
#endif
}

void Graph::FreeHalfGraph() {
#ifdef CUDA
  if (halfEdges_) {
    CUDA_TRY(cudaFree(halfEdges_));
    halfEdges_ = nullptr;
  }
  if (halfNodeIndex_) {
    CUDA_TRY(cudaFree(halfNodeIndex_));
    halfNodeIndex_ = nullptr;
  }
  if (halfAdj_) {
    CUDA_TRY(cudaFree(halfAdj_));
    halfAdj_ = nullptr;
  }
#else
  if (halfEdges_) {
    MyFree((void *&)halfEdges_, halfEdgesNum_ * sizeof(uint64_t));
  }
  if (halfNodeIndex_) {
    MyFree((void *&)halfNodeIndex_, (halfNodesNum_ + 1) * sizeof(EdgeT));
  }
  if (halfAdj_) {
    MyFree((void *&)halfAdj_, halfEdgesNum_ * sizeof(NodeT));
  }
#endif
}

Graph::~Graph() {
  FreeRawGraph();
  FreeGraph();
  FreeHalfGraph();
}

NodeT Graph::GetMaxCore() {
  log_info(coreClock_.Start());

  ::ConstructCSRGraph(cudaRawEdges_, rawEdgesNum_, rawNodeIndex_, rawAdj_);
  log_info(coreClock_.Count("Construct Raw CSR Graph"));

  NodeT maxCoreNum = ::KCore(rawNodeIndex_, rawAdj_, rawNodesNum_, rawCore_);
  log_info(coreClock_.Count("KCore maxCoreNum: %u", maxCoreNum));

  return maxCoreNum;
}

// 获取max-k-truss主流程
NodeT Graph::KMaxTruss(NodeT startK, NodeT startLevel) {
  // 预处理
  Preprocess(startK);
  if (edgesNum_ == 0) {
    return 0;
  }

  // 三角形计数
  log_info(triCountClock_.Start());
#ifdef CUDA
  ::GetEdgeSup(nodeIndex_, adj_, edgesId_, nodesNum_, edgesSup_);
#else
  ::GetEdgeSup(halfNodeIndex_, halfAdj_, halfNodesNum_, edgesSup_);
#endif
  log_info(triCountClock_.Count("Count"));

  // TODO can remove
  uint64_t count = 0;
  for (uint64_t i = 0; i < halfEdgesNum_; i++) {
    count += edgesSup_[i];
  }
  log_info(triCountClock_.Count("triangle count: %lu", count / 3));

  // 求解k-truss
  log_info(trussClock_.Start());
#ifdef CUDA
  ::KTruss(nodeIndex_, adj_, edgesId_, nodesNum_, halfEdges_, halfEdgesNum_, edgesSup_, startLevel);
#else
  ::KTruss(nodeIndex_, adj_, edgesId_, halfEdges_, halfEdgesNum_, edgesSup_, startLevel);
#endif
  log_info(trussClock_.Count("KTruss"));

  FreeHalfGraph();
  FreeGraph();

  // 打印信息
  NodeT possibleKMax = DisplayStats(edgesSup_, halfEdgesNum_, startK);

  return possibleKMax;
}

// 图的预处理
void Graph::Preprocess(NodeT startK) {
  log_info(preprocessClock_.Start());
  log_info(preprocessClock_.Count("startK: %u", startK));

  if (startK > 2u) {
    edgesNum_ = ::ConstructNewGraph(cudaRawEdges_, edges_, rawCore_, rawEdgesNum_, startK);
    log_info(
        preprocessClock_.Count("edgesNum_: %u %.2fMB", edgesNum_, (double)edgesNum_ * sizeof(uint64_t) / 1024 / 1024));
    if (edgesNum_ == 0) {
      return;
    }
  } else {
#ifdef CUDA
    // cuda memcpy
    edges_ = cudaRawEdges_;
    cudaRawEdges_ = nullptr;
#else
    edges_ = rawEdges_;
    rawEdges_ = nullptr;
#endif
    edgesNum_ = rawEdgesNum_;
  }

  ::ConstructCSRGraph(edges_, edgesNum_, nodeIndex_, adj_);
  nodesNum_ = FIRST(edges_[edgesNum_ - 1]) + 1;
  log_info(preprocessClock_.Count("CSR nodesNum_: %u", nodesNum_));

  halfEdgesNum_ = edgesNum_ / 2;
  ::ConstructHalfEdges(edges_, halfEdges_, halfEdgesNum_);
  log_info(preprocessClock_.Count("halfEdgesNum_: %u", halfEdgesNum_));

  ::ConstructCSRGraph(halfEdges_, halfEdgesNum_, halfNodeIndex_, halfAdj_);
  halfNodesNum_ = FIRST(halfEdges_[halfEdgesNum_ - 1]) + 1;
  log_info(preprocessClock_.Count("Half CSR halfNodesNum_: %u", halfNodesNum_));

  ::GetEdgesId(edges_, edgesNum_, halfNodeIndex_, halfAdj_, edgesId_);
  log_info(preprocessClock_.Count("GetEdgesId"));
}
