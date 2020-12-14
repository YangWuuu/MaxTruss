CXX=g++
CXXFLAGS_SERIAL=-O3 -std=c++14 -Iinclude
CXXFLAGS=-O3 -std=c++14 -fopenmp -Iinclude
CUDAFLAGS=-O3 -std=c++14 -fopenmp -Iinclude -I/usr/local/cuda/include -DCUDA
NVCC=nvcc
NVCCFLAGS=-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_60,code=sm_60 -O3 -std=c++14 -Iinclude -Xcompiler -fopenmp
SRC_FILES=src/main.cpp src/log.cpp src/read_file.cpp src/graph.cpp src/ktruss.cpp src/kcore.cpp src/preprocess.cpp src/tricount.cpp src/util.cpp

all : kmax_truss_omp kmax_truss_serial kron_gen

kmax_truss_omp :
	$(CXX) $(CXXFLAGS) ${SRC_FILES} -o $@

kmax_truss_cuda : graph.o log.o main.o read_file.o util.o gpu.o kcore_cu.o preprocess_cu.o tricount_cu.o ktruss_cu.o
	$(CXX) $(CUDAFLAGS) $^ -L /usr/local/cuda/lib64 -lcudart -o $@

%.o : src/%.cpp
	$(CXX) $(CUDAFLAGS) -c $^ -o $@

kcore_cu.o : cuda/kcore.cu
	$(NVCC) $(NVCCFLAGS) -dc $^ -o $@

preprocess_cu.o : cuda/preprocess.cu
	$(NVCC) $(NVCCFLAGS) -dc $^ -o $@

tricount_cu.o : cuda/tricount.cu
	$(NVCC) $(NVCCFLAGS) -dc $^ -o $@

ktruss_cu.o : cuda/ktruss.cu
	$(NVCC) $(NVCCFLAGS) -dc $^ -o $@

gpu.o : kcore_cu.o preprocess_cu.o tricount_cu.o ktruss_cu.o
	$(NVCC) $(NVCCFLAGS) -dlink $^ -o $@

kmax_truss_serial :
	$(CXX) $(CXXFLAGS_SERIAL) ${SRC_FILES} -DSERIAL -o $@

kron_gen :
	$(CXX) $(CXXFLAGS) src/kron_gen.cpp src/log.cpp -o $@

.PHONY : clean
clean :
	rm kmax_truss_omp kmax_truss_serial kron_gen
