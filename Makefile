CXX=g++
CXXFLAGS_SERIAL=-O3 -std=c++11 -Iinclude
CXXFLAGS=-O3 -std=c++11 -fopenmp -Iinclude
NVCC=nvcc
NVCCFLAGS=-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_60,code=sm_60 -O3 -std=c++11 -Iinclude -Xcompiler -fopenmp
SRC_FILES=src/main.cpp src/log.cpp src/read_file.cpp src/graph.cpp src/ktruss.cpp src/kcore.cpp src/preprocess.cpp src/tricount.cpp src/util.cpp

all : kmax_truss_omp kmax_truss_serial kron_gen

kmax_truss_omp :
	$(CXX) $(CXXFLAGS) ${SRC_FILES} -o $@

#kmax_truss_cuda : graph.o kcore.o ktruss.o log.o main.o preprocess.o read_file.o gpu.o tricount_cu.o
#	$(CXX) $(CXXFLAGS) $^ -L /usr/local/cuda/lib64 -lcudart -o $@
#
#%.o : src/%.cpp
#	$(CXX) $(CXXFLAGS) -c $^ -o $@
#
#tricount_cu.o : cuda/tricount.cu
#	$(NVCC) $(NVCCFLAGS) -dc $^ -o $@
#
#gpu.o : tricount_cu.o
#	$(NVCC) $(NVCCFLAGS) -dlink $^ -o $@

kmax_truss_serial :
	$(CXX) $(CXXFLAGS_SERIAL) ${SRC_FILES} -DSERIAL -o $@

kron_gen :
	$(CXX) $(CXXFLAGS) src/kron_gen.cpp src/log.cpp -o $@

.PHONY : clean
clean :
	rm kmax_truss_omp kmax_truss_serial kron_gen
