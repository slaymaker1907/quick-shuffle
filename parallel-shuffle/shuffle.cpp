#include <cstdio>
#include <random>
#include <chrono>
#include <algorithm>
#include <new>
#include <vector>
#include <utility>

#include "HeapSet.hpp"
#include "ThreadPool.hpp"

std::random_device rd;
std::minstd_rand gen(rd());

template<typename NumberT>
NumberT divceil(NumberT a, NumberT b) {
    NumberT result = a / b;
    return (result * b) < a ? result + 1 : result;
}

template<typename T>
void fisher_yates(T *data, size_t size, std::mt19937_64 *rng) {
    //printf("This partition has %zu items...\n", arr.size());
    size_t size_bound = size - 1;
    for (size_t i = 0; i < size_bound; i++) {
        std::uniform_int_distribution<size_t> dis(i, size_bound);
        size_t swap_with = dis(*rng);
        std::swap(data[i], data[swap_with]);
    }
}

#define T int
//template<typename T>
void assign_partition(T *input,
                      size_t size,
                      size_t pcount,
                      cuda_permute::HeapSet<T> *partitions,
                      std::mt19937_64 rng) {
    auto **node_buffers = (cuda_permute::HeapSetNode<T> **)malloc(sizeof(cuda_permute::HeapSetNode<T>*) * pcount);
    assert(node_buffers);
    for (size_t p = 0; p < pcount; p++) {
        node_buffers[p] = partitions[p].new_node();
    }
    size_t buffer_capacity = partitions[0].get_node_size();

    std::uniform_int_distribution<size_t> dis(0, pcount-1);
    for (size_t i = 0; i < size; i++) {
        size_t partition = dis(rng);

        cuda_permute::HeapSetNode<T> *buffer = node_buffers[partition];
        if (buffer->size() >= buffer_capacity) {
            auto full_part = partitions + partition;
            full_part->add_to_size(buffer->size());
            buffer = full_part->new_node();
            node_buffers[partition] = buffer;
        }

        buffer->push_back(input[i]);
    }
    free(node_buffers);
}

//template<typename T>
void shuffle_partition(cuda_permute::HeapSet<T> *input, T *output, std::mt19937_64 rng) {
    size_t size = input->get_size();
    input->move_to_buffer(output);
    fisher_yates(output, size, &rng);
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

template<typename T>
void parallel_shuffle(T *input, size_t size, size_t seed = 8675309) {
    size_t thread_count = std::thread::hardware_concurrency();
    ThreadPool pool(thread_count);
    size_t pcount = size_sqrt(size);
    size_t part_block_size = size_sqrt(pcount);

    cuda_permute::HeapSet<T> *partitions = (cuda_permute::HeapSet<T>*)malloc(sizeof(cuda_permute::HeapSet<T>) * pcount);
    assert(partitions);
    for (size_t part = 0; part < pcount; part++) {
        new(partitions + part) cuda_permute::HeapSet<T>(part_block_size);
    }

    std::mt19937_64 base_generator(seed);
    // std::mtr19937_64 *thread_rngs = malloc(sizeof(std::mtr19937_64) * pcount);
    // assert(thread_rngs);
    // for (size_t i = 0; i < thread_count; i++) {
    //     new(thread_rngs + i) std::mtr19937_64(base_generator());
    // }

    std::vector<std::future<void>> partition_futures(0);
    partition_futures.reserve(thread_count);

    size_t eles_per_thread = divceil(size, (size_t)thread_count);
    size_t unpartitioned = size;
    T *unpartitioned_start = input;

    for (size_t i = 0; i < thread_count && unpartitioned > 0; i++) {
        size_t to_partition = std::min(unpartitioned, eles_per_thread);
        std::mt19937_64 thread_rng(base_generator);
        // assign_partition(unpartitioned_start, to_partition, pcount, partitions, thread_rng);
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
        std::mt19937_64 part_rng(base_generator);
        // shuffle_partition(partition, output, part_rng);
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

    free(input);

    return 0;
}
