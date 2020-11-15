CXX=g++
CXXFLAGS=-O3 -std=c++11 -fopenmp -Iinclude
SRC_FILES=src/main.cpp src/log.cpp src/read_file.cpp src/graph.cpp

all : kmax_truss kmax_truss_serial

kmax_truss :
	$(CXX) $(CXXFLAGS) ${SRC_FILES} -o $@

kmax_truss_serial :
	$(CXX) $(CXXFLAGS) ${SRC_FILES} -D SERIAL -o $@

.PHONY : clean
clean :
	rm kmax_truss kmax_truss_serial
