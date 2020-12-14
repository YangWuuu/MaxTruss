#include <omp.h>

#include <string>
#include <thread>

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
#ifdef CUDA
  log_info("cuda run. thread num: %d", omp_get_max_threads());
#else
  log_info("omp run. thread num: %d", omp_get_max_threads());
#endif
#endif

  std::string arg_string;
  for (int i = 0; i < argc; i++) {
    arg_string += std::string(argv[i]) + " ";
  }
  log_info("argc: %d argv is %s", argc, arg_string.c_str());
}

int main(int argc, char *argv[]) {
  Log(argc, argv);

#ifdef CUDA
  // cuda runtime 第一次启动比较慢
  std::thread t([]() {
    log_info("cuda runtime init");
    cudaSetDevice(0);
    log_info("cuda runtime end");
  });
#endif

  if (argc < 3) {
    fprintf(stderr, "usage: %s -f graph_name\n", argv[0]);
    exit(-1);
  }
  std::string filePath = std::string(argv[2]);

  Clock allClock("All");
  log_info(allClock.Start());

  // 文件读取
  uint64_t *edges{nullptr};
  ReadFile readFile(filePath);
  uint64_t edgesNum = readFile.ConstructEdges(edges);

#ifdef CUDA
  t.join();
#endif

  Graph graph(edges, edgesNum);
  // 通过Core分解获取最大的层次信息
  NodeT maxCore = graph.GetMaxCore();

  // 得到kmax上界
  NodeT kMaxUpperBound = maxCore + 2;
  //  NodeT kMaxUpperBound = 0;
  // 得到kmax下界
  NodeT kMaxLowerBound = graph.KMaxTruss(kMaxUpperBound, 0u);
  log_info("UpperBound: %u LowerBound: %u", kMaxUpperBound, kMaxLowerBound);

  if (kMaxLowerBound < kMaxUpperBound) {
    // 得到kmax真实值
    NodeT kMax = graph.KMaxTruss(kMaxLowerBound, kMaxLowerBound - 2);
    log_info("LowerBound: %u kMax: %u", kMaxLowerBound, kMax);
    if (kMax < kMaxLowerBound) {
      log_error("it is error");
      exit(-1);
    }
  }

  log_info(allClock.Count("End"));
  return 0;
}
