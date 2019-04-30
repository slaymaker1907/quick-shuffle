#pragma once

#include <atomic>
#include <vector>
#include <assert.h>
#include <cstdlib>
#include <cstring>

namespace cuda_permute
{

template<typename T>
class HeapSet;

template<typename T>
class HeapSetNode {
    friend class HeapSet<T>;
private:
    T *_data;
    HeapSetNode<T> *_next;
    size_t _size;

    HeapSetNode(T *data, HeapSetNode<T> *next = nullptr) {
        assert(data);
        _data = data;
        _next = next;
        _size = 0;
    }
public:
    size_t size() {
        return _size;
    }
    void push_back(T element) {
        _data[_size] = element;
        _size++;
    }
    T* data() {
        return _data;
    }
};

template<typename T>
class HeapSet {
private:
    // // disable copy constructor so we don't accidently use it.
    // HeapSet(const HeapSet<T> &);
    // HeapSet& operator=(const HeapSet &);

    std::atomic<HeapSetNode<T>*> head;
    std::atomic<size_t> size;
    size_t node_size;

public:
    HeapSet(size_t _node_size) {
        // make sure we are using lock free stuff, otherwise why bother?
        assert(head.is_lock_free());
        assert(size.is_lock_free());

        head.store(nullptr, std::memory_order::memory_order_relaxed);
        size.store(0, std::memory_order::memory_order_relaxed);
        node_size = _node_size;

        atomic_thread_fence(std::memory_order::memory_order_release);
    }
    // HeapSet(HeapSet &&) noexcept = default;
    // HeapSet& operator=(HeapSet &&) noexcept = default;
    // ~HeapSet() {
    //     HeadSetNode<T> *current = head.load(std::memory_order::memory_order_acquire);
    //     while (current) {
    //         HeadSetNode<T> *next = current->next;
    //         free(current);
    //         current = next;
    //     }
    // }

    size_t get_size() {
        return size.load(std::memory_order_consume);
    }

    size_t get_node_size() {
        return node_size;
    }

    HeapSetNode<T>* new_node() {
        auto *node = (HeapSetNode<T>*)malloc(sizeof(HeapSetNode<T>) + sizeof(T) * node_size);
        assert(node);
        T* data_start = (T*)(node + 1);
        HeapSetNode<T>* old_head = head.exchange(node, std::memory_order::memory_order_acq_rel);
        new(node) HeapSetNode<T>(data_start, old_head);
        return node;
    }

    size_t add_to_size(size_t to_add) {
        return size.fetch_add(to_add, std::memory_order::memory_order_relaxed);
    }

    // this only needs to be sequentially correct now.
    // buffer size should be >= size.
    void move_to_buffer(T *buffer) {
        HeapSetNode<T> *current = head.load(std::memory_order::memory_order_acquire);
        while (current) {
            memcpy(buffer, current->_data, current->_size * sizeof(T));
            buffer += current->_size;
            HeapSetNode<T> *next = current->_next;
            free(current);
            current = next;
        }
    }

    size_t true_size() {
        size_t true_size = 0;
        HeapSetNode<T> *current = head.load(std::memory_order::memory_order_acquire);
        while (current) {
            true_size += current->_size;
            HeapSetNode<T> *next = current->_next;
            current = next;
        }

        return true_size;
    }
};

}
