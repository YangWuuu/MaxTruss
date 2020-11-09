#include <string>

#include "clock.h"
#include "log.h"
#include "read_file.h"
#include "util.h"

#pragma ide diagnostic ignored "openmp-use-default-none"

int main(int argc, char *argv[]) {
  std::string logFlag = std::string(argv[argc - 1]);
  if (logFlag == "debug") {
    log_set_level(LOG_DEBUG);
  } else if (logFlag == "info") {
    log_set_level(LOG_INFO);
  } else {
    log_set_quiet(true);
  }
  std::string arg_string;
  for (int i = 0; i < argc; i++) {
    arg_string += std::string(argv[i]) + " ";
  }
  log_info("argc: %d argv is %s", argc, arg_string.c_str());

  std::string filePath = GetFileName(argc, argv);

  Clock allClock("All");
  Clock readFileClock("ReadFile");
  Clock preprocessClock("Preprocess");
  Clock trussClock("Truss");

  log_info(allClock.Start());

  log_info(readFileClock.Start());
  uint64_t *edges{nullptr};
  ReadFile readFile(filePath, readFileClock);
  EdgeT edgesNum = readFile.ConstructEdges(edges);
  log_info(readFileClock.Count("End"));

  log_info(preprocessClock.Start());
  auto *edgesFirst = (NodeT *)malloc(edgesNum * sizeof(NodeT));
  auto *edgesSecond = (NodeT *)malloc(edgesNum * sizeof(NodeT));
  for (EdgeT i = 0; i < edgesNum; i++) {
    edgesFirst[i] = FIRST(edges[i]);
    edgesSecond[i] = SECOND(edges[i]);
  }
  log_info(preprocessClock.Count("Unzip"));

  NodeT nodesNum = edgesFirst[edgesNum - 1] + 1;
  log_info(preprocessClock.Count("nodesNum: %llu", nodesNum));

  EdgeT halfEdgesNum = edgesNum / 2;
  auto *halfEdges = (uint64_t *)malloc(halfEdgesNum * sizeof(uint64_t));
  std::copy_if(edges, edges + edgesNum, halfEdges,
               [](const uint64_t &edge) { return FIRST(edge) < SECOND(edge); });
  log_info(preprocessClock.Count("halfEdgesNum: %llu", halfEdgesNum));

  auto *deg = (NodeT *)calloc(nodesNum, sizeof(NodeT));
  log_info(preprocessClock.Count("deg calloc"));
  for (EdgeT i = 0; i < edgesNum; i++) {
    ++deg[edgesFirst[i]];
  }
  log_info(preprocessClock.Count("deg"));
  auto *nodeIndex = (EdgeT *)calloc((nodesNum + 1), sizeof(EdgeT));
  log_info(preprocessClock.Count("nodeIndex calloc"));
  for (NodeT i = 0; i < nodesNum; i++) {
    nodeIndex[i + 1] = nodeIndex[i] + deg[i];
  }
  log_info(preprocessClock.Count("nodeIndex"));

  auto *edgesId = (EdgeT *)malloc(edgesNum * sizeof(EdgeT));
  //  GetEdgesId(edgesId, nodeIndex, edgesSecond, nodesNum, preprocessClock);
  GetEdgesId2(edgesId, nodeIndex, edgesSecond, nodesNum, edgesNum,
              preprocessClock);
  log_info(preprocessClock.Count("edgesId"));

  log_info(trussClock.Start());

  auto *edgesSup = (EdgeT *)calloc(halfEdgesNum, sizeof(EdgeT));
  GetEdgeSup(edgesSup, nodeIndex, edgesSecond, edgesId, nodesNum);
  log_info(trussClock.Count("GetEdgeSup"));

  uint64_t count = 0;
  for (uint64_t i = 0; i < halfEdgesNum; i++) {
    count += edgesSup[i];
  }
  log_info(trussClock.Count("triangle count: %lu", count / 3));

  KTruss(nodeIndex, edgesSecond, edgesSup, edgesId, halfEdgesNum, halfEdges);
  log_info(trussClock.Count("KTruss"));

  displayStats(edgesSup, halfEdgesNum);
  log_info(trussClock.Count("displayStats"));

  log_info(allClock.Count("End"));
  return 0;
}
