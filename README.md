# 大规模图数据中kmax-truss问题的求解和算法优化

> [比赛网址](https://www.datafountain.cn/competitions/473/datasets)

## 代码编译说明

* 方法一：在代码目录下运行`make`命令，当前目录得到`kmax_truss`可执行程序。

* 方法二：

```shell script
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

在项目根目录下的bin目录内得到`kmax_truss`可执行程序

## 运行使用说明

运行命令：

```shell script
./kmax_truss -f ***.tsv(图数据文件路径)
```

标准输出为：

```text
kmax = x, Edges in kmax-truss = y.
```

其中x、y为整数，x代表kmax-truss值，y代表kmax-truss中的边数量。

## 基本算法介绍

主要分成四个部分：

* 读取文件
* 数据预处理
* 三角形记数
* Truss降维

## 并行化设计思路和方法

### 读取文件

采用多线程分块读取文件，自动识别块首和块尾。

### 数据预处理

边的解压，CSR建图，获取边的序号等操作均采取并行化设计。

### 三角形记数

采用并行点迭代的方式，获取每条边的三角形数量。

### Truss降维

支持边数量刷新，边的筛选均采用并行计算。

## 算法优化

所有部分都是并行计算的。

## 详细算法设计与实现

### 读取文件

核心代码如下所示：

```cpp
  std::vector<std::thread> threads(FILE_SPLIT_NUM);
  for (uint64_t i = 0; i < FILE_SPLIT_NUM; i++) {
    uint64_t start = len_ * i / FILE_SPLIT_NUM;
    uint64_t end = len_ * (i + 1) / FILE_SPLIT_NUM;
    if (i != 0) {
      while (*(byte_ + start) != '\n') {
        ++start;
      }
      ++start;
    }
    if (i + 1 != FILE_SPLIT_NUM) {
      while (*(byte_ + end) != '\n') {
        ++end;
      }
      ++end;
    }
    threads[i] = std::thread(
        [=]() { ::GetEdges(edges + edgesNum, byte_ + start, end - start); });
    edgesNum += ::GetLineNum(byte_ + start, end - start);
  }
  for (auto &thread : threads) {
    thread.join();
  }
```

### 数据预处理

```c++
void Unzip(const uint64_t *edges, EdgeT edgesNum, NodeT *&edgesFirst,
           NodeT *&edgesSecond) {
  edgesFirst = (NodeT *)malloc(edgesNum * sizeof(NodeT));
  edgesSecond = (NodeT *)malloc(edgesNum * sizeof(NodeT));
#pragma omp parallel for
  for (EdgeT i = 0; i < edgesNum; i++) {
    edgesFirst[i] = FIRST(edges[i]);
    edgesSecond[i] = SECOND(edges[i]);
  }
}
```

```c++
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
```

### 三角形记数

```c++
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
```

## 实验结果与分析

### 图数据集

> [下载地址](http://datafountain.int-yt.com/Files/BDCI2020/473HuaKeDaKtruss/ktruss-data.zip)

| **数据集**               | **说明**                          | **kmax** | **Edges in kmax-truss** |
|:---------------------:|:-----------------------------------:|:--------:|:-----------------------:|
| s18.e16.rmat.edgelist | 顶点数：0.2 million、边数：7 million  | 164      | 225,529                 |
| s19.e16.rmat.edgelist | 顶点数：0.5 million、边数：15 million | 223      | 334,934                 |
| cit-Patents           | 顶点数：3.7 million、边数：33 million | 36       | 2,625                   |
| soc-LiveJournal       | 顶点数：4.8 million、边数：85 million | 362      | 72,913                  |

以上文件中，图用顺序存放的边表表示。每条边包含两个顶点以及权重（其中源顶点和目标顶点各占4个字节），边的存储长度为8个字节，文件的存储格式如下：  
目标顶点（4字节，无符号整型） 源顶点（4字节，无符号整型） 权重  
目标顶点（4字节，无符号整型） 源顶点（4字节，无符号整型） 权重  
……[EOF]  

### 运行环境

4核8线程 2.6GHz

```shell script
# cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c
      8  Intel(R) Xeon(R) Gold 6278C CPU @ 2.60GHz
# cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l
1
# cat /proc/cpuinfo| grep "cpu cores"| uniq
cpu cores       : 4
# cat /proc/cpuinfo| grep "processor"| wc -l
8
```

### 实验结果

| **数据集**            | **单线程(ms)** | **多线程(ms)** |
|:---------------------:|:-----------:|:-----------:|
| s18.e16.rmat.edgelist | 27662       | 3542        |
| s19.e16.rmat.edgelist | 67733       | 8648        |
| cit-Patents           | 2451        | 587         |
| soc-LiveJournal       | 33635       | 4844        |

## 程序代码模块说明

| ****文件****                           | ****功能****         |
|----------------------------------------|--------------------|
| [src/main.cpp](src/main.cpp)           | 代码主流程              |
| [src/log.cpp](src/log.cpp)             | 日志打印               |
| [src/read_file.cpp](src/read_file.cpp) | 文件读取               |
| [src/graph.cpp](src/graph.cpp)         | 图预处理，三角形记数，Truss降维 |

