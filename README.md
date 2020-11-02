# 大规模图数据中kmax-truss问题的求解和算法优化

> [比赛网址](https://www.datafountain.cn/competitions/473/datasets)

## 代码编译说明
在代码目录下运行`make`命令，当前目录得到`kmax_truss`可执行程序。

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
TODO

## 并行化设计思路和方法
TODO

## 算法优化
TODO

## 详细算法设计与实现
TODO

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
| s18.e16.rmat.edgelist | 48073       | 6337        |
| s19.e16.rmat.edgelist | 134172      | 17316       |
| cit-Patents           | 17559       | 3362        |
| soc-LiveJournal       | 134535      | 19733       |



## 程序代码模块说明
TODO
