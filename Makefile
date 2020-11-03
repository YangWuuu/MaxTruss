CXX=g++
CXXFLAGS=-O3 -std=c++11 -fopenmp -Iinclude

all : kmax_truss

kmax_truss :
	$(CXX) $(CXXFLAGS) src/main.cpp src/log.cpp -o kmax_truss

.PHONY : clean
clean :
	rm kmax_truss
