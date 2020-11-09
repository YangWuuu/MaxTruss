#pragma once

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "clock.h"
#include "log.h"

#pragma ide diagnostic ignored "openmp-use-default-none"

#define FIRST(x) (uint32_t(x >> 32u))
#define SECOND(x) (uint32_t(x & 0xFFFFFFFF))
#define MAKEEDGE(x, y) ((uint64_t)(uint64_t(x) << 32u | uint64_t(y)))

using NodeT = uint32_t;
using EdgeT = uint32_t;

const int CACHE_LINE_ENTRY = 16;

inline std::string GetFileName(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s -f graph_name\n", argv[0]);
    exit(-1);
  }
  return std::string(argv[2]);
}

void GetEdgesId(EdgeT *edgesId, const EdgeT *nodeIndex,
                const NodeT *edgesSecond, NodeT nodesNum,
                Clock &preprocessClock) {
  auto *nodeIndexCopy = (EdgeT *)malloc((nodesNum + 1) * sizeof(EdgeT));
  for (NodeT i = 0; i < nodesNum + 1; i++) {
    nodeIndexCopy[i] = nodeIndex[i];
  }
  log_info(preprocessClock.Count("nodeIndexCopy"));
  EdgeT edgeId = 0;
  for (NodeT u = 0; u < nodesNum; u++) {
    for (EdgeT j = nodeIndex[u]; j < nodeIndex[u + 1]; j++) {
      NodeT v = edgesSecond[j];
      if (u < v) {
        edgesId[j] = edgeId;
        nodeIndexCopy[u]++;
        if (edgesSecond[nodeIndexCopy[v]] == u) {
          edgesId[nodeIndexCopy[v]] = edgeId;
          nodeIndexCopy[v]++;
        }
        edgeId++;
      }
    }
  }
}

/*
 * InclusivePrefixSumOMP: General Case Inclusive Prefix Sum
 * histogram: is for cache-aware thread-local histogram purpose
 * output: should be different from the variables captured in function object f
 * size: is the original size for the flagged prefix sum
 * f: requires it as the parameter, f(it) return the histogram value of that it
 */
template <typename H, typename T, typename F>
void InclusivePrefixSumOMP(std::vector<H> &histogram, T *output, size_t size,
                           F f) {
  int omp_num_threads = omp_get_num_threads();

#pragma omp single
  { histogram = std::vector<H>((omp_num_threads + 1) * CACHE_LINE_ENTRY, 0); }
  static thread_local int tid = omp_get_thread_num();
  // 1st Pass: Histogram.
  auto avg = size / omp_num_threads;
  auto it_beg = avg * tid;
  auto histogram_idx = (tid + 1) * CACHE_LINE_ENTRY;
  histogram[histogram_idx] = 0;
  auto it_end = tid == omp_num_threads - 1 ? size : avg * (tid + 1);
  size_t prev = 0u;
  for (auto it = it_beg; it < it_end; it++) {
    auto value = f(it);
    histogram[histogram_idx] += value;
    prev += value;
    output[it] = prev;
  }
#pragma omp barrier

  // 2nd Pass: single-prefix-sum & Add previous sum.
#pragma omp single
  {
    for (auto local_tid = 0; local_tid < omp_num_threads; local_tid++) {
      auto local_histogram_idx = (local_tid + 1) * CACHE_LINE_ENTRY;
      auto prev_histogram_idx = (local_tid)*CACHE_LINE_ENTRY;
      histogram[local_histogram_idx] += histogram[prev_histogram_idx];
    }
  }
  {
    auto prev_sum = histogram[tid * CACHE_LINE_ENTRY];
    for (auto it = it_beg; it < it_end; it++) {
      output[it] += prev_sum;
    }
#pragma omp barrier
  }
}

template <typename T>
uint32_t BranchFreeBinarySearch(const T *a, const uint32_t offset_beg,
                                const uint32_t offset_end, T x) {
  int32_t n = offset_end - offset_beg;
  using I = uint32_t;
  const T *base = a + offset_beg;
  while (n > 1) {
    I half = n / 2;
    __builtin_prefetch(base + half / 2, 0, 0);
    __builtin_prefetch(base + half + half / 2, 0, 0);
    base = (base[half] < x) ? base + half : base;
    n -= half;
  }
  return (*base < x) + base - a;
}

// Assuming (offset_beg != offset_end)
template <typename T>
uint32_t GallopingSearch(const T *array, const uint32_t offset_beg,
                         const uint32_t offset_end, T val) {
  if (array[offset_end - 1] < val) {
    return offset_end;
  }
  // galloping
  if (array[offset_beg] >= val) {
    return offset_beg;
  }
  if (array[offset_beg + 1] >= val) {
    return offset_beg + 1;
  }
  if (array[offset_beg + 2] >= val) {
    return offset_beg + 2;
  }

  auto jump_idx = 4u;
  while (true) {
    auto peek_idx = offset_beg + jump_idx;
    if (peek_idx >= offset_end) {
      return BranchFreeBinarySearch(array, (jump_idx >> 1u) + offset_beg + 1,
                                    offset_end, val);
    }
    if (array[peek_idx] < val) {
      jump_idx <<= 1u;
    } else {
      return array[peek_idx] == val
                 ? peek_idx
                 : BranchFreeBinarySearch(array,
                                          (jump_idx >> 1u) + offset_beg + 1,
                                          peek_idx + 1, val);
    }
  }
}

template <typename T>
uint32_t LinearSearch(const T *array, const uint32_t offset_beg,
                      const uint32_t offset_end, T val) {
  // linear search fallback
  for (auto offset = offset_beg; offset < offset_end; offset++) {
    if (array[offset] >= val) {
      return offset;
    }
  }
  return offset_end;
}

inline NodeT FindSrc(const EdgeT *nodeIndex, NodeT nodesNum, NodeT u,
                     const EdgeT edgeIdx) {
  if (edgeIdx >= nodeIndex[u + 1]) {
    // update last_u, preferring galloping instead of binary search because not
    // large range here
    u = GallopingSearch(nodeIndex, static_cast<uint32_t>(u) + 1,
                        static_cast<uint32_t>(nodesNum + 1), edgeIdx);
    // 1) first > , 2) has neighbor
    if (nodeIndex[u] > edgeIdx) {
      while (nodeIndex[u] - nodeIndex[u - 1] == 0) {
        u--;
      }
      u--;
    } else {
      // g->num_edges[u] == i
      while (nodeIndex[u + 1] - nodeIndex[u] == 0) {
        u++;
      }
    }
  }
  return u;
}

void GetEdgesId2(EdgeT *edgesId, const EdgeT *nodeIndex,
                 const NodeT *edgesSecond, NodeT nodesNum, EdgeT edgesNum,
                 Clock &preprocessClock) {
  auto *nodeIndexCopy = (EdgeT *)malloc((nodesNum + 1) * sizeof(EdgeT));
  nodeIndexCopy[0] = 0;
  // Edge upper_tri_start of each edge
  auto *upper_tri_start = (EdgeT *)malloc(nodesNum * sizeof(EdgeT));

  auto num_threads = omp_get_max_threads();
  std::vector<uint32_t> histogram(CACHE_LINE_ENTRY * num_threads);

#pragma omp parallel
  {
#pragma omp for
    // Histogram (Count).
    for (NodeT u = 0; u < nodesNum; u++) {
      upper_tri_start[u] =
          (nodeIndex[u + 1] - nodeIndex[u] > 256)
              ? GallopingSearch(edgesSecond, nodeIndex[u], nodeIndex[u + 1], u)
              : LinearSearch(edgesSecond, nodeIndex[u], nodeIndex[u + 1], u);
#ifdef SEQ_SCAN
      num_edges_copy[u + 1] = nodeIndex[u + 1] - upper_tri_start[u];
#endif
    }

    // Scan.
    InclusivePrefixSumOMP(histogram, nodeIndexCopy + 1, nodesNum, [&](int u) {
      return nodeIndex[u + 1] - upper_tri_start[u];
    });

    // Transform.
    NodeT u = 0;
#pragma omp for schedule(dynamic, 6000)
    for (EdgeT j = 0u; j < edgesNum; j++) {
      u = FindSrc(nodeIndex, nodesNum, u, j);
      if (j < upper_tri_start[u]) {
        auto v = edgesSecond[j];
        auto offset = BranchFreeBinarySearch(edgesSecond, nodeIndex[v],
                                             nodeIndex[v + 1], u);
        auto eid = nodeIndexCopy[v] + (offset - upper_tri_start[v]);
        edgesId[j] = eid;
      } else {
        edgesId[j] = nodeIndexCopy[u] + (j - upper_tri_start[u]);
      }
    }
  }
  free(upper_tri_start);
  free(nodeIndexCopy);
}

void GetEdgeSup(NodeT *edgesSup, const EdgeT *nodeIndex,
                const NodeT *edgesSecond, const EdgeT *edgesId,
                NodeT nodesNum) {
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

void KTruss(const EdgeT *nodeIndex, const NodeT *edgesSecond, EdgeT *edgesSup,
            const EdgeT *edgesId, EdgeT halfEdgesNum,
            const uint64_t *halfEdges) {
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

void displayStats(const EdgeT *EdgeSupport, EdgeT halfEdgesNum) {
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
  printf("kmax = %u, Edges in kmax-truss = %u.\n", maxSup + 2,
         numEdgesWithMaxSup);
}
