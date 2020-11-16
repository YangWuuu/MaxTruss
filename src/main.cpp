#include <omp.h>

#include <string>

#include "clock.h"
#include "graph.h"
#include "log.h"
#include "read_file.h"
#include "util.h"

#pragma ide diagnostic ignored "openmp-use-default-none"

int main(int argc, char *argv[]) {
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

  std::string filePath = GetFileName(argc, argv);

  Clock allClock("All");
  log_info(allClock.Start());

  // 文件读取
  uint64_t *edges{nullptr};
  ReadFile readFile(filePath);
  uint64_t edgesNum = readFile.ConstructEdges(edges);

  // 求解max-k-truss
  Graph graph(edges, edgesNum);
  if (!graph.MaxKTruss(true)) {
    graph.MaxKTruss(false);
  }

  log_info(allClock.Count("End"));
  return 0;
}
