CXX=g++
CXXFLAGS=-std=c++11 -Wall -Wextra -O3 #-fsanitize=address
#CXXFLAGS=-std=c++11 -Wall -Wextra -Og -g
LINKERS=-pthread
# LINKER=

all: benchmark

benchmark: shuffle.cpp ThreadPool.hpp
	$(CXX) $< $(CXXFLAGS) -o $@ $(LINKERS)

clean:
	rm -rf shuffle shuffle.exe benchmark benchmark.exe

