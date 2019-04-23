CXX=g++
CXXFLAGS=-std=c++11 -Wall -Wextra -O3
LINKERS=-ltbb -pthread

all: shuffle

shuffle: shuffle.cpp ThreadPool.hpp
	$(CXX) $< $(CXXFLAGS) -o $@ $(LINKERS)

clean:
	rm -rf shuffle

