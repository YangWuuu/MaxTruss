#include <string>

#include "clock.h"
#include "graph.h"
#include "log.h"
#include "read_file.h"
#include "util.h"

void Log(int argc, char *argv[]) {
  // 打印日志
  std::string logFlag = std::string(argv[argc - 1]);
  if (logFlag == "debug") {
    log_set_level(LOG_DEBUG);
  } else if (logFlag == "info") {
    log_set_level(LOG_INFO);
  } else {
    log_set_quiet(true);
  }

#ifdef SERIAL
  log_info("serial run");
#else
  log_info("parallel run. thread num: %d", omp_get_max_threads());
#endif

  std::string arg_string;
  for (int i = 0; i < argc; i++) {
    arg_string += std::string(argv[i]) + " ";
  }
  log_info("argc: %d argv is %s", argc, arg_string.c_str());
}

int main(int argc, char *argv[]) {
  Log(argc, argv);

  std::string filePath = GetFileName(argc, argv);

  Clock allClock("All");
  log_info(allClock.Start());

  // 文件读取
  uint64_t *edges{nullptr};
  ReadFile readFile(filePath);
  uint64_t edgesNum = readFile.ConstructEdges(edges);

  // 求解max-k-truss
  Graph graph(edges, edgesNum);
  // TODO Graph中内存需要统一全局分配
  NodeT maxK = graph.GetMaxK();

  NodeT startK = maxK - 1;

  while (true) {
    NodeT possibleKMax1 = graph.MaxKTruss(startK);
    log_info("startK: %u possibleKMax1: %u", startK, possibleKMax1);
    if (possibleKMax1 >= startK) {
      break;
    }

    startK = possibleKMax1;
    NodeT possibleKMax2 = graph.MaxKTruss(startK);
    log_info("maxK: %u possibleKMax2: %u", startK, possibleKMax2);
    if (possibleKMax2 >= startK) {
      break;
    }

    // can not go here
    startK = 0;
    NodeT possibleKMax3 = graph.MaxKTruss(startK);
    log_info("maxK: %u possibleKMax3: %u", startK, possibleKMax3);
    if (possibleKMax3 >= startK) {
      break;
    }
  }

  log_info(allClock.Count("End"));
  return 0;
}
