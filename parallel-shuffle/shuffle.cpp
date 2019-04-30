#include <cstdio>
#include <random>
#include <chrono>
#include <algorithm>
#include <new>
#include <vector>
#include <utility>

#include "HeapSet.hpp"
#include "ThreadPool.hpp"
#include "FastRandRange.hpp"

#define CCAST(expr, type) ((type)expr)
#define RNG_T std::mt19937

template<typename T>
double inversion_percentage(T *data, size_t size) {
    size_t pair_count = 0;
    size_t inverted = 0;
    for (size_t i1 = 0; i1 < size; i1++) {
        T ele1 = data[i1];
        for (size_t i2 = i1+1; i2 < size; i2++) {
            T ele2 = data[i2];
            ++pair_count;
            if (ele2 < ele1) {
                ++inverted;
            }
        }
    }

    return CCAST(inverted, double) / CCAST(pair_count, double);
}

template<typename NumberT>
NumberT divceil(NumberT a, NumberT b) {
    NumberT result = a / b;
    return (result * b) < a ? result + 1 : result;
}

template<typename T>
void fisher_yates(T *data, size_t size, RNG_T *rng) {
    //printf("This partition has %zu items...\n", arr.size());
    size_t size_bound = size - 1;
    for (size_t i = 0; i < size_bound; i++) {
        FastRandRange dis(i, size);
        size_t swap_with = dis.rand_in_range(rng);
        // assert(swap_with >= i && swap_with < size);
        std::swap(data[i], data[swap_with]);
    }
}

#define T int
//template<typename T>
void assign_partition(T *input,
                      size_t size,
                      size_t pcount,
                      cuda_permute::HeapSet<T> *partitions,
                      RNG_T rng) {
    auto **node_buffers = (cuda_permute::HeapSetNode<T> **)malloc(sizeof(cuda_permute::HeapSetNode<T>*) * pcount);
    assert(node_buffers);
    for (size_t p = 0; p < pcount; p++) {
        node_buffers[p] = partitions[p].new_node();
    }
    size_t buffer_capacity = partitions[0].get_node_size();

    // std::uniform_int_distribution<size_t> dis(0, pcount-1);
    FastRandRange dis(0, pcount);
    for (size_t i = 0; i < size; i++) {
        size_t partition = dis.rand_in_range(&rng);

        cuda_permute::HeapSetNode<T> *buffer = node_buffers[partition];
        if (buffer->size() >= buffer_capacity) {
            auto full_part = partitions + partition;
            full_part->add_to_size(buffer->size());
            buffer = full_part->new_node();
            node_buffers[partition] = buffer;
        }

        buffer->push_back(input[i]);
    }
    for (size_t p = 0; p < pcount; p++) {
        auto full_part = partitions + p;
        full_part->add_to_size(node_buffers[p]->size());
    }
    free(node_buffers);
}

//template<typename T>
void shuffle_partition(cuda_permute::HeapSet<T> *input, T *output, RNG_T rng) {
    size_t size = input->get_size();
    //if (size > 0) printf("%zu\n", size);
    input->move_to_buffer(output);
    if (size > 1) {
        fisher_yates(output, size, &rng);
    }
}
#undef T

int* generate_input(size_t size) {
    int *result = (int*)malloc(size * sizeof(int));
    assert(result);
    for (size_t i = 0; i < size; i++) {
        result[i] = i;
    }
    return result;
}

size_t size_sqrt(size_t n) {
    return (size_t)std::ceil(sqrt(n));
}

int size_log2(size_t n) {
    int result = 0;
    while (n > 0) {
        n /= 2;
        result++;
    }
    return result;
}

template<typename T>
void parallel_shuffle(T *input, size_t size, int seed = 8675309) {
    size_t thread_count = std::thread::hardware_concurrency();
    ThreadPool pool(thread_count);
    size_t pcount = thread_count;
    size_t part_block_size = divceil(size, pcount) / 8;

    cuda_permute::HeapSet<T> *partitions = (cuda_permute::HeapSet<T>*)malloc(sizeof(cuda_permute::HeapSet<T>) * pcount);
    assert(partitions);
    for (size_t part = 0; part < pcount; part++) {
        new(partitions + part) cuda_permute::HeapSet<T>(part_block_size);
    }

    RNG_T base_generator(seed);

    std::vector<std::future<void>> partition_futures(0);
    partition_futures.reserve(thread_count);

    size_t eles_per_thread = divceil(size, (size_t)thread_count);
    size_t unpartitioned = size;
    T *unpartitioned_start = input;

    for (size_t i = 0; i < thread_count && unpartitioned > 0; i++) {
        size_t to_partition = std::min(unpartitioned, eles_per_thread);
        RNG_T thread_rng(base_generator());
        partition_futures.emplace_back(pool.enqueue(assign_partition, unpartitioned_start, to_partition, pcount, partitions, thread_rng));

        unpartitioned -= to_partition;
        unpartitioned_start += to_partition;
    } 

    for (auto &future : partition_futures) {
        future.get();
    }
    partition_futures.clear();

    T *output = input;

    for (size_t p = 0; p < pcount; p++) {
        cuda_permute::HeapSet<T> *partition = partitions + p;
        RNG_T part_rng(base_generator());
        partition_futures.emplace_back(pool.enqueue(shuffle_partition, partition, output, part_rng));
        output += partition->get_size();
    }

    for (auto &future : partition_futures) {
        future.get();
    }

    // free(thread_rngs);
    free(partitions);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: ./shuffle <n>\n");
        return -1;
    }

    printf("Generating the input vector...\n");
    size_t n = std::atoll(argv[1]);
    int *input = generate_input(n);

 
    auto begin = std::chrono::high_resolution_clock::now();
    parallel_shuffle(input, n);
    auto end = std::chrono::high_resolution_clock::now();
    printf("Shuffle takes: %.6fs\n", std::chrono::duration<double>(end - begin).count());
    // printf("Percent of pairs inverted: %f\n", inversion_percentage(input, n));
    
    free(input);

    return 0;
}
