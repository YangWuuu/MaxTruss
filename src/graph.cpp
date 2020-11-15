#include "graph.h"

#include <algorithm>
#include <map>

#pragma ide diagnostic ignored "openmp-use-default-none"

bool Graph::MaxKTruss(bool remove) {
  Preprocess(remove);

  TriCount();

  log_info(trussClock.Start());
  ::KTruss(nodeIndex_, edgesSecond_, edgesId_, halfEdges_, halfEdgesNum_,
           edgesSup_);
  log_info(trussClock.Count("KTruss"));

  bool isValid = displayStats(edgesSup_, halfEdgesNum_, minK_);
  log_info(trussClock.Count("displayStats isValid: %d", isValid));

  return isValid;
}

void Graph::Preprocess(bool remove) {
  log_info(preprocessClock.Start());

  if (!repeat_) {
    rawDeg_ = (NodeT *)calloc(rawNodesNum_, sizeof(NodeT));
    ::CalDeg(rawEdges_, rawEdgesNum_, rawDeg_);
    log_info(preprocessClock.Count("cal deg"));
    repeat_ = true;
  }

  if (remove) {
    RemoveEdges();
    nodesNum_ = FIRST(edges_[edgesNum_ - 1]) + 1;
    log_info(preprocessClock.Count("nodesNum_: %u", nodesNum_));
    deg_ = (NodeT *)calloc(nodesNum_, sizeof(NodeT));
    ::CalDeg(edges_, edgesNum_, deg_);
    log_info(preprocessClock.Count("cal deg"));
  } else {
    minK_ = 0;
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

void Graph::RemoveEdges() {
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
  minK_ = maxK / SHRINK_SIZE;
  log_info(preprocessClock.Count("minK: %u maxK: %u", minK_, maxK));

  edges_ = (uint64_t *)malloc(rawEdgesNum_ * sizeof(uint64_t));
  // TODO parallel
  edgesNum_ = std::copy_if(rawEdges_, rawEdges_ + rawEdgesNum_, edges_,
                           [&](const uint64_t edge) {
                             return rawDeg_[FIRST(edge)] > minK_ &&
                                    rawDeg_[SECOND(edge)] > minK_;
                           }) -
              edges_;

  log_info(preprocessClock.Count("edgesNum_: %u", edgesNum_));
}

void Graph::TriCount() {
  log_info(triCountClock.Start());
  edgesSup_ = (EdgeT *)calloc(halfEdgesNum_, sizeof(EdgeT));
  GetEdgeSup(nodeIndex_, edgesSecond_, edgesId_, nodesNum_, edgesSup_);
  log_info(triCountClock.Count("Count"));

  // TODO can remove
  uint64_t count = 0;
  for (uint64_t i = 0; i < halfEdgesNum_; i++) {
    count += edgesSup_[i];
  }
  log_info(triCountClock.Count("triangle count: %lu", count / 3));
}

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
  // Edge upper_tri_start of each edge
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
#ifdef SEQ_SCAN
      num_edges_copy[u + 1] = nodeIndex[u + 1] - upper_tri_start[u];
#endif
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

void NodeIndex(const NodeT *deg, NodeT nodesNum, EdgeT *&nodeIndex) {
  nodeIndex = (EdgeT *)calloc((nodesNum + 1), sizeof(EdgeT));
  // TODO parallel
  for (NodeT i = 0; i < nodesNum; i++) {
    nodeIndex[i + 1] = nodeIndex[i] + deg[i];
  }
}

void GetEdgeSup(const EdgeT *nodeIndex, const NodeT *edgesSecond,
                const EdgeT *edgesId, NodeT nodesNum, NodeT *edgesSup) {
  auto *startEdge = (EdgeT *)malloc(nodesNum * sizeof(EdgeT));
  for (NodeT i = 0; i < nodesNum; i++) {
    EdgeT j = nodeIndex[i];
    EdgeT endIndex = nodeIndex[i + 1];

    while (j < endIndex) {
      if (edgesSecond[j] > i) break;
      j++;
    }
    startEdge[i] = j;
  }
#pragma omp parallel
  {
    auto *X = (EdgeT *)calloc(nodesNum, sizeof(EdgeT));
#pragma omp for schedule(dynamic, 64)
    for (NodeT u = 0; u < nodesNum; u++) {
      for (EdgeT j = startEdge[u]; j < nodeIndex[u + 1]; j++) {
        NodeT w = edgesSecond[j];
        X[w] = j + 1;
      }

      for (EdgeT j = nodeIndex[u]; j < startEdge[u]; j++) {
        NodeT v = edgesSecond[j];

        for (EdgeT k = nodeIndex[v + 1] - 1; k >= startEdge[v]; k--) {
          NodeT w = edgesSecond[k];
          // check if: w > u
          if (w <= u) {
            break;
          }

          if (X[w]) {  // This is a triangle
            // edge id's are: <u,w> : g->eid[ X[w] -1]
            //<u,w> : g->eid[ X[w] -1]
            //<v,u> : g->eid[ j ]
            //<v,w> : g->eid[ k ]
            EdgeT e1 = edgesId[X[w] - 1], e2 = edgesId[j], e3 = edgesId[k];
            __sync_fetch_and_add(&edgesSup[e1], 1);
            __sync_fetch_and_add(&edgesSup[e2], 1);
            __sync_fetch_and_add(&edgesSup[e3], 1);
          }
        }
      }

      for (EdgeT j = startEdge[u]; j < nodeIndex[u + 1]; j++) {
        NodeT w = edgesSecond[j];
        X[w] = 0;
      }
    }
#pragma omp barrier
  }
}

void Scan(EdgeT numEdges, const EdgeT *edgesSup, int level, EdgeT *curr,
          EdgeT &currTail, bool *inCurr) {
  // Size of cache line
  const EdgeT BUFFER_SIZE_BYTES = 2048;
  const EdgeT BUFFER_SIZE = BUFFER_SIZE_BYTES / sizeof(NodeT);

  NodeT buff[BUFFER_SIZE];
  EdgeT index = 0;

#pragma omp for schedule(static)
  for (long i = 0; i < numEdges; i++) {
    if (edgesSup[i] == level) {
      buff[index] = i;
      inCurr[i] = true;
      index++;

      if (index >= BUFFER_SIZE) {
        long tempIdx = __sync_fetch_and_add(&currTail, BUFFER_SIZE);

        for (long j = 0; j < BUFFER_SIZE; j++) {
          curr[tempIdx + j] = buff[j];
        }
        index = 0;
      }
    }
  }

  if (index > 0) {
    long tempIdx = __sync_fetch_and_add(&currTail, index);

    for (EdgeT j = 0; j < index; j++) {
      curr[tempIdx + j] = buff[j];
    }
  }

#pragma omp barrier
}

void SubLevel(const EdgeT *nodeIndex, const NodeT *edgesSecond,
              const EdgeT *curr, bool *inCurr, EdgeT currTail, EdgeT *edgesSup,
              int level, EdgeT *next, bool *inNext, EdgeT &nextTail,
              bool *processed, const EdgeT *edgesId,
              const uint64_t *halfEdges) {
  // Size of cache line
  const long BUFFER_SIZE_BYTES = 2048;
  const long BUFFER_SIZE = BUFFER_SIZE_BYTES / sizeof(NodeT);

  NodeT buff[BUFFER_SIZE];
  EdgeT index = 0;

#pragma omp for schedule(dynamic, 4)
  for (long i = 0; i < currTail; i++) {
    // process edge <u,v>
    EdgeT e1 = curr[i];
    NodeT u = FIRST(halfEdges[e1]);
    NodeT v = SECOND(halfEdges[e1]);

    EdgeT uStart = nodeIndex[u], uEnd = nodeIndex[u + 1];
    EdgeT vStart = nodeIndex[v], vEnd = nodeIndex[v + 1];

    unsigned int numElements = (uEnd - uStart) + (vEnd - vStart);
    EdgeT j_index = uStart, k_index = vStart;

    for (unsigned int innerIdx = 0; innerIdx < numElements; innerIdx++) {
      if (j_index >= uEnd) {
        break;
      } else if (k_index >= vEnd) {
        break;
      } else if (edgesSecond[j_index] == edgesSecond[k_index]) {
        EdgeT e2 = edgesId[k_index];  //<v,w>
        EdgeT e3 = edgesId[j_index];  //<u,w>

        // If e1, e2, e3 forms a triangle
        if ((!processed[e2]) && (!processed[e3])) {
          // Decrease support of both e2 and e3
          if (edgesSup[e2] > level && edgesSup[e3] > level) {
            // Process e2
            int supE2 = __sync_fetch_and_sub(&edgesSup[e2], 1);
            if (supE2 == (level + 1)) {
              buff[index] = e2;
              inNext[e2] = true;
              index++;
            }

            if (supE2 <= level) {
              __sync_fetch_and_add(&edgesSup[e2], 1);
            }

            if (index >= BUFFER_SIZE) {
              long tempIdx = __sync_fetch_and_add(&nextTail, BUFFER_SIZE);

              for (long bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++)
                next[tempIdx + bufIdx] = buff[bufIdx];
              index = 0;
            }

            // Process e3
            int supE3 = __sync_fetch_and_sub(&edgesSup[e3], 1);

            if (supE3 == (level + 1)) {
              buff[index] = e3;
              inNext[e3] = true;
              index++;
            }

            if (supE3 <= level) {
              __sync_fetch_and_add(&edgesSup[e3], 1);
            }

            if (index >= BUFFER_SIZE) {
              long tempIdx = __sync_fetch_and_add(&nextTail, BUFFER_SIZE);

              for (long bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++)
                next[tempIdx + bufIdx] = buff[bufIdx];
              index = 0;
            }
          } else if (edgesSup[e2] > level) {
            // process e2 only if e1 < e3
            if (e1 < e3 && inCurr[e3]) {
              int supE2 = __sync_fetch_and_sub(&edgesSup[e2], 1);

              if (supE2 == (level + 1)) {
                buff[index] = e2;
                inNext[e2] = true;
                index++;
              }

              if (supE2 <= level) {
                __sync_fetch_and_add(&edgesSup[e2], 1);
              }

              if (index >= BUFFER_SIZE) {
                long tempIdx = __sync_fetch_and_add(&nextTail, BUFFER_SIZE);

                for (long bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++)
                  next[tempIdx + bufIdx] = buff[bufIdx];
                index = 0;
              }
            }
            if (!inCurr[e3]) {  // if e3 is not in curr array then decrease
              // support of e2
              int supE2 = __sync_fetch_and_sub(&edgesSup[e2], 1);
              if (supE2 == (level + 1)) {
                buff[index] = e2;
                inNext[e2] = true;
                index++;
              }

              if (supE2 <= level) {
                __sync_fetch_and_add(&edgesSup[e2], 1);
              }

              if (index >= BUFFER_SIZE) {
                long tempIdx = __sync_fetch_and_add(&nextTail, BUFFER_SIZE);

                for (long bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++)
                  next[tempIdx + bufIdx] = buff[bufIdx];
                index = 0;
              }
            }
          } else if (edgesSup[e3] > level) {
            // process e3 only if e1 < e2
            if (e1 < e2 && inCurr[e2]) {
              int supE3 = __sync_fetch_and_sub(&edgesSup[e3], 1);

              if (supE3 == (level + 1)) {
                buff[index] = e3;
                inNext[e3] = true;
                index++;
              }

              if (supE3 <= level) {
                __sync_fetch_and_add(&edgesSup[e3], 1);
              }

              if (index >= BUFFER_SIZE) {
                long tempIdx = __sync_fetch_and_add(&nextTail, BUFFER_SIZE);

                for (long bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++)
                  next[tempIdx + bufIdx] = buff[bufIdx];
                index = 0;
              }
            }
            if (!inCurr[e2]) {  // if e2 is not in curr array then decrease
              // support of e3
              int supE3 = __sync_fetch_and_sub(&edgesSup[e3], 1);

              if (supE3 == (level + 1)) {
                buff[index] = e3;
                inNext[e3] = true;
                index++;
              }

              if (supE3 <= level) {
                __sync_fetch_and_add(&edgesSup[e3], 1);
              }

              if (index >= BUFFER_SIZE) {
                long tempIdx = __sync_fetch_and_add(&nextTail, BUFFER_SIZE);

                for (long bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++)
                  next[tempIdx + bufIdx] = buff[bufIdx];
                index = 0;
              }
            }
          }
        }
        j_index++;
        k_index++;
      } else if (edgesSecond[j_index] < edgesSecond[k_index]) {
        j_index++;
      } else if (edgesSecond[k_index] < edgesSecond[j_index]) {
        k_index++;
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

#pragma omp for schedule(static)
  for (long i = 0; i < currTail; i++) {
    EdgeT e = curr[i];

    processed[e] = true;
    inCurr[e] = false;
  }

#pragma omp barrier
}

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

  // parallel region
#pragma omp parallel
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
  }  // End of parallel region

  // Free memory
  free(next);
  free(inNext);
  free(curr);
  free(inCurr);
  free(processed);
}

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

  for (long i = 0; i < halfEdgesNum; i++) {
    if (EdgeSupport[i] == minSup) {
      numEdgesWithMinSup++;
    }

    if (EdgeSupport[i] == maxSup) {
      numEdgesWithMaxSup++;
    }
  }

  std::vector<uint64_t> sups(maxSup + 1);
  for (long i = 0; i < halfEdgesNum; i++) {
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
