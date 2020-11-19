// this code rewrite from
// https://github.com/wang-zhq/kron_generate/blob/master/kron_generate_4.cpp

#include <fcntl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>

#include "log.h"
#include "util.h"

#pragma ide diagnostic ignored "openmp-use-default-none"

const uint32_t seed = 2020u;

int main(int argc, char const **argv) {
  if (argc < 3) {
    log_error("./kron_gen [点参数] [边参数]");
    exit(-1);
  }

  std::string arg_string;
  for (int i = 0; i < argc; i++) {
    arg_string += std::string(argv[i]) + " ";
  }
  log_info("argc: %d argv is %s", argc, arg_string.c_str());

  const uint64_t scaleFactor = std::stoi(argv[1]);
  const uint64_t edgeFactor = std::stoi(argv[2]);
  const std::string fileName =
      "s" + std::string(argv[1]) + ".e" + std::string(argv[2]) + ".tsv";

  auto nodesNum = static_cast<uint64_t>(pow(2., scaleFactor));
  uint64_t edgesNum = nodesNum * edgeFactor;

  log_info("nodes: %u edges: %u", nodesNum, edgesNum);

  auto *vertex = new uint32_t[2 * edgesNum];

#pragma omp parallel for
  for (uint64_t i = 0; i < 2 * edgesNum; i++) {
    vertex[i] = 1;
  }

  double const a = 0.57;
  double const b = 0.19;
  double const c = 0.19;

  double ab = a + b;
  double cNorm = c / (1 - ab);
  double aNorm = a / ab;

  std::default_random_engine e(seed);
  std::uniform_real_distribution<double> u(0, 1);

  for (uint64_t k = 0; k < scaleFactor; k++) {
    auto kPow = static_cast<uint64_t>(pow(2., k));

    for (int64_t i = 0; i < edgesNum; i++) {
      bool hBit = (u(e) > ab);
      bool fBit = (u(e) > (cNorm * hBit + aNorm * (!hBit)));

      vertex[2 * i] += kPow * hBit;
      vertex[2 * i + 1] += kPow * fBit;
    }
  }

  auto *randPerm = new uint32_t[nodesNum];
#pragma omp parallel for
  for (uint64_t i = 0; i < nodesNum; i++) {
    randPerm[i] = i;
  }

  shuffle(randPerm, randPerm + nodesNum, e);

#pragma omp parallel for
  for (uint64_t i = 0; i < 2 * edgesNum; i++) {
    vertex[i] = randPerm[vertex[i]];
  }

  log_info("gen data finish");
  auto *edges = new uint64_t[2 * edgesNum];
  auto *halfEdges = reinterpret_cast<uint64_t *>(vertex);
#pragma omp parallel for
  for (uint64_t i = 0; i < edgesNum; i++) {
    edges[2 * i] = halfEdges[i];
    edges[2 * i + 1] = MAKE_EDGE(SECOND(halfEdges[i]), FIRST(halfEdges[i]));
  }
  edgesNum *= 2;
  log_info("edgesNum: %u", edgesNum);

  std::sort(edges, edges + edgesNum);
  log_info("sort edgesNum: %u", edgesNum);

  edgesNum = std::unique(edges, edges + edgesNum) - edges;
  log_info("unique edgesNum: %u", edgesNum);

  edgesNum = std::remove_if(edges, edges + edgesNum,
                            [](const uint64_t edge) {
                              return FIRST(edge) == SECOND(edge);
                            }) -
             edges;
  log_info("remove self loop edgesNum: %u", edgesNum);

  std::string result;
  for (uint64_t i = 0; i < edgesNum; i++) {
    result += std::to_string(SECOND(edges[i]));
    result += "\t";
    result += std::to_string(FIRST(edges[i]));
    result += "\t1\n";
  }

  FILE *fp = fopen(fileName.c_str(), "w+");
  fwrite(result.c_str(), result.size(), 1, fp);
  fclose(fp);

  delete[] vertex;
  delete[] edges;
  delete[] randPerm;

  log_info("%s all finish", fileName.c_str());
  return 0;
}
