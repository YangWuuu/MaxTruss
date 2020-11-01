CXX=g++
CXXFLAGS=-O2 -std=c++11 -fopenmp -Iinclude

all : truss

truss : main.o log.o
	$(CXX) $(CXXFLAGS) -o truss main.o log.o

main.o : src/main.cpp
	$(CXX) $(CXXFLAGS) -c src/main.cpp -o main.o

log.o : src/log.cpp
	$(CXX) $(CXXFLAGS) -c src/log.cpp -o log.o

.PHONY : clean
clean :
	rm truss *.o

