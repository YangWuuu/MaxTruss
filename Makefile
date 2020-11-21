CXX=g++
CXXFLAGS=-O3 -std=c++11 -fopenmp -Iinclude
SRC_FILES=src/main.cpp src/log.cpp src/read_file.cpp src/graph.cpp src/ktruss.cpp src/kcore.cpp

all : kmax_truss kmax_truss_serial kron_gen

kmax_truss :
	$(CXX) $(CXXFLAGS) ${SRC_FILES} -o $@

kmax_truss_serial :
	$(CXX) $(CXXFLAGS) ${SRC_FILES} -D SERIAL -o $@

kron_gen :
	$(CXX) $(CXXFLAGS) src/kron_gen.cpp src/log.cpp -o $@

.PHONY : clean
clean :
	rm kmax_truss kmax_truss_serial kron_gen
