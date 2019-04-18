#ifndef UNROLLED_LIST
#define UNROLLED_LIST

#include <cstdint>
#include "cuda_polyfill.hpp"

namespace unrolled_list
{

template <class T>
class UnrolledListNode {
    UnrolledListNode<T> *next_node;
    int size;
    T data[0];
};

// This class supports concurrency for the creation stage, but not for the iteration stage.
// Additionally, it is designed to work only on the GPU.
// It may allocate more memory than is strictly necessary to avoid synchronization, though it will try to
// avoid this.
template <class T>
class UnrolledList {
private:
    volatile UnrolledListNode<T> *unfilled_block_list;
    volatile UnrolledListNode<T> *first_block;
    int block_size;
    int first_block_size;
public:
    UnrolledList(int block_size) {
        this.block_size = block_size;
    }
    void push_many(T *data, int data_size) {
        atomicCAS(
    }
};

}
#endif
