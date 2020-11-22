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

  // TODO Graph中内存需要统一全局分配
  Graph graph(edges, edgesNum);

  NodeT maxCore = graph.GetMaxCore();

  NodeT kMaxUpperBound = maxCore + 2;
  NodeT kMaxLowerBound = graph.KMaxTruss(kMaxUpperBound, 0u);
  log_info("UpperBound: %u LowerBound: %u", kMaxUpperBound, kMaxLowerBound);

  if (kMaxLowerBound < kMaxUpperBound) {
    NodeT kMax = graph.KMaxTruss(kMaxLowerBound, kMaxLowerBound - 2);
    //    NodeT kMax = graph.KMaxTruss(kMaxLowerBound, 0);
    //    NodeT kMax = graph.KMaxTruss(0, 0);
    log_info("LowerBound: %u kMax: %u", kMaxLowerBound, kMax);
    if (kMax < kMaxLowerBound) {
      log_error("it is error");
      exit(-1);
    }
  }

  log_info(allClock.Count("End"));
  return 0;
}
