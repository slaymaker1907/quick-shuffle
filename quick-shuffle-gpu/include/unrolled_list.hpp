#ifndef UNROLLED_LIST
#define UNROLLED_LIST

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <assert.h>
#include <cmath>
#include "cuda_polyfill.hpp"
#include "cuda_ptr_cas.hpp"
#include "permute_util.hpp"

namespace cuda_permute
{

template <class T>
struct UnrolledListNode {
    UnrolledListNode<T> *next_node;
    volatile UnrolledListNode<T> *next_unfilled_node;
    volatile int size; // volatile should be unnecessary due to the way we use CAS.
    volatile int reader_count;
    T data[];
};

// This class supports concurrency for the creation stage, but not for the iteration stage.
// Additionally, it is designed to work only on the GPU.
// It may allocate more memory than is strictly necessary to avoid synchronization, though it will try to
// avoid this.
// This class weak fairness such that if a thread continuously wants to be executed, it will eventually be executed regardless
// of the halting behavior of other threads.
template <class T>
class UnrolledList {
public:
    volatile UnrolledListNode<T> *unfilled_block_list;
    volatile UnrolledListNode<T> *first_block;
    int block_size;
    volatile int threadc_finding_node;

    UnrolledList(int input_block_size) {
        block_size = input_block_size;
        unfilled_block_list = nullptr;
        first_block = nullptr;
        threadc_finding_node = 0;
    }

    __device__ bool is_empty() {
        return unfilled_block_list == nullptr;
    }

    __device__ int node_count() {
        int result = 0;
        UnrolledListNode<T> *current = (UnfilledListNode<T>*)unfilled_block_list;
        while (current != nullptr) {
            result++;
            current = current->next_node;
        }
        return result;
    }

    __device__ void push_many(T *data, int data_size) {
        while (data_size > 0) {
            UnrolledListNode<T> *insert_into = nullptr;
            // First try to find an existing node with empty slots.
            while (true) {
                UnrolledListNode<T> *expected = (UnrolledListNode<T>*)unfilled_block_list;
                if (expected == nullptr) {
                    break;
                }
                UnrolledListNode<T> *next_node = expected->next_node;
                if (cuda_ptr_cas((UnrolledListNode<T>**)&unfilled_block_list, expected, next_node) == expected) {
                    insert_into = expected;
                    break;
                }
            }

            // If no node was found, create a new one.
            if (insert_into == nullptr) {
                insert_into = (UnrolledListNode<T>*)malloc(sizeof(UnrolledListNode<T>) + sizeof(T) * block_size);
                assert(insert_into);
                insert_into->next_unfilled_node = nullptr;
                insert_into->size = 0;
                insert_into->reader_count = 0;

                // First insert it into the main list.
                while (true) {
                    UnrolledListNode<T> *full_list_head = (UnrolledListNode<T>*)first_block;
                    insert_into->next_node = full_list_head;

                    // fence here to prevent threads from reading it without being initialized.
                    __threadfence();
                    if (cuda_ptr_cas((UnrolledListNode<T>**)&first_block, full_list_head, insert_into) == full_list_head) {
                        break;
                    }
                }

                // Now, insert it into the list of unfilled blocks.
                while (true) {
                    UnrolledListNode<T> *unfilled_list_head = (UnrolledListNode<T>*)unfilled_block_list;
                    insert_into->next_unfilled_node = unfilled_list_head;

                    // fence here to prevent threads from reading it without being initialized.
                    __threadfence();
                    if (cuda_ptr_cas((UnrolledListNode<T>**)&unfilled_block_list, unfilled_list_head, insert_into)
                        == unfilled_list_head) {
                        break;
                    }
                }
            }

            // Don't know how much we can put in the list, so we have to check using atomics.
            int old_size = atomicAdd((int*)&insert_into->size, data_size);
            int move_size = data_size;
            int new_size = old_size + move_size;
            if (new_size > block_size) {
                move_size = block_size - old_size;
                insert_into->size = block_size;
            }
            memcpy(&insert_into->data[old_size], data, sizeof(T) * move_size);
            data_size -= move_size;
            data += move_size;
        }
    }

    // pops up to max_size elements and moves them to buffer, returing number of copied elements.
    // this function guarantees that it will return 0 iff this list is empty.
    // assumes max_size > 0 as a precondition.
    __device__ int pop_to_buffer(T *buffer, int max_size) {
        // guarantee safe memory free by having the last reader free and by
        // popping from list then not calling free until there are no threads potentially looking at the
        // head of the list.

        int to_copy = max_size;
        int start_from;

        // first find a non-empty node if one exists. by adding one to threadc_finding_node,
        // we guarantee that until we decrement threadc_finding_node, no node will be deallocated.
        atomicAdd((int*)&threadc_finding_node, 1);
        __threadfence(); // must guarantee that other threads see our increment before we try to read anything.
        UnrolledListNode<T> *head_node = (UnrolledListNode<T>*)first_block;
        while (head_node != nullptr) {
            int old_size = atomicSub((int*)&head_node->size, max_size);
            if (old_size > 0) {
                // found a node, insert ourselves into the readers and remove ourselves from the finding_node set.
                // reader variable is necessary to avoid freeing while still copying data from straggler.
                atomicAdd((int*)&head_node->reader_count, 1);

                // the reader count MUST be incremented before decrementing threadc_finding_node
                // since if both are zero, it is safe to free the node.
                __threadfence();

                // since we are in the reader set, can remove ourselves from the finding_node set.
                atomicSub((int*)&threadc_finding_node, 1);

                if (max_size >= old_size) {
                    // head_node->size may be negative, but that is ok.
                    start_from = 0;
                    to_copy = old_size;
                } else {
                    // max_size < old_size.
                    start_from = old_size - max_size;
                }
                break;

            } else {
                head_node = head_node->next_node;
            }
        }

        // check to see if we ever found a node to copy from.
        if (head_node == nullptr) {
            // don't need a fence here since we don't read from global memory.
            // __threadfence();
            atomicSub((int*)&threadc_finding_node, 1);
            return 0;
        }

        memcpy(buffer, &head_node->data[start_from], to_copy * sizeof(T));

        __threadfence(); // make sure our copy completes before we decrement reader.
        atomicSub((int*)&head_node->reader_count, 1);

        // If we were the last reader to decrement size, we need to deallocate eventually.
        if (start_from == 0) {
            // remove from list so no one else can read this node.
            while (cuda_ptr_cas(&first_block, head_node, head_node->next_node) == head_node) {
                // keep trying till we succeed in removing the node.
                // this may be stalled since we might not be the true head of the list.
            }

            // make sure cas happens before we wait for global list readers
            __threadfence();

            // wait for anyone finding a node to read from.
            while (threadc_finding_node > 0) {
                // busy wait
            }
            // even if a new thread comes in reading list, it doesn't matter since 
            // they will see the list without this node.

            // wait for other readers too finish.
            // don't need a fence between the head_node->reader_count and threadc_finding_node
            // since they are independent.
            while (head_node->reader_count > 0) {
                // busy wait.
            }

            // only this thread may now access this node.
            // since this is c-based, don't need to worry about original size of node.
            free(head_node);
        }

        // as a post condition, getting here guarantees that to_copy > 0.
        return to_copy;
    }
};

template <class T>
UnrolledList<T>* move_to_gpu(T const* input, size_t size) {
    double dbsqrt = sqrt((double)size);
    int block_size = (int)ceil(dbsqrt);
    assert(block_size > 0);

    size_t node_size = sizeof(UnrolledListNode<T>) + sizeof(T) * block_size;
    UnrolledListNode<T> *host_node = (UnrolledListNode<T>*)malloc(node_size);
    assert(host_node);
    host_node->next_node = nullptr;
    host_node->next_unfilled_node = nullptr;
    host_node->reader_count = 0;
    host_node->size = 0;

    UnrolledListNode<T> *last_device_node = nullptr;

    CudaStream stream1, stream2;

    UnrolledListNode *device_node_cudaMalloc;
    check_cuda_error(cudaMalloc((void**)&device_node, node_size));

    UnrolledListNode **page_locked_node;
    page_locked_node = alloc_pinned(page_locked_node);

    // TODO: fix this by making sure we only call malloc on the GPU.
    while (size > 0) {
        int to_move = (int)std::min((size_t)block_size, size);
        host_node->size = to_move;
        host_node->next_node = last_device_node;
        memcpy(&host_node->data[0], input, to_move * sizeof(T));

        stream1.memcpy_to_device(CCAST(device_node_cudaMalloc, char*), CCAST(host_node, char*), node_size);
        malloc_on_gpu(page_locked_node, 1, &stream2, node_size);

        stream1.join();
        stream2.join();

        cudaMalloc_to_malloc(*page_locked_node, device_node_cudaMalloc, 1, &stream2, node_size);

        last_device_node = *page_locked_node;
        size -= to_move;
        input = input + to_move;
    }

    free_pinned(page_locked_node);

    // now finally create the actual list.
    UnrolledList<T> host_list(block_size);
    UnrolledList<T> *device_list;

    host_list.first_block = last_device_node;
    if (host_node->size < block_size) {
        host_list.unfilled_block_list = last_device_node;
    }

    check_cuda_error(cudaMalloc((void**)&device_list, sizeof(UnrolledList<T>)));
    stream1.memcpy_to_device(device_list, &host_list);
    check_cuda_error(cudaDeviceSynchronize());

    free(host_node);

    return device_list;
}


template <class T>
void copy_list_to_cpu(T *output, UnrolledList *input, size_t size) {
    double dbsqrt = sqrt((double)size);
    int output_buffer_size = (int)ceil(dbsqrt);
    assert(output_buffer_size > 0);

    while (size > 0) {
    }
}

}
#endif
