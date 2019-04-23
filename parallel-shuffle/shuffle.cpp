#include <cstdio>
#include <random>
#include <chrono>
#include <algorithm>

#include <tbb/concurrent_vector.h>
#include "ThreadPool.hpp"

using namespace tbb;

std::random_device rd;
std::minstd_rand gen(rd());

void fisher_yates(concurrent_vector<size_t>& arr) {
    //printf("This partition has %zu items...\n", arr.size());
    for (size_t i = arr.size() - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dis(0, i);
        size_t j = dis(gen);
        size_t t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
    }
}

void assign_partition(size_t arr[], size_t length, 
                      concurrent_vector<size_t> parts[],
                      size_t n_parts) {
    std::uniform_int_distribution<size_t> dis(0, n_parts - 1);
    for (size_t i = 0; i < length; ++i) {
        parts[dis(gen)].push_back(arr[i]);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: ./shuffle <n>\n");
        return -1;
    }

    printf("Generating the input vector...\n");
    size_t n = std::atoll(argv[1]);
    size_t* data = new size_t[n];
    std::uniform_int_distribution<size_t> dis(0, 200000000);
    for (size_t i = 0; i < n; ++i) data[i] = dis(gen);
 
    ThreadPool pool(std::thread::hardware_concurrency());
    size_t n_parts = (size_t)std::ceil(std::sqrt(n));
    size_t part_size = (size_t)(std::ceil((double)n / n_parts));
    printf("n_parts: %zu, part_size: %zu\n", n_parts, part_size);
    auto parts = new concurrent_vector<size_t>[n_parts];
    for (size_t i = 0; i < n_parts; ++i) {
        parts[i].reserve(part_size);
    }

    auto begin = std::chrono::high_resolution_clock::now();
    std::vector<std::future<void>> res;
    for (size_t i = 0; i < n_parts; ++i) {
        auto arr = data + i * part_size;
        auto len = (i == n_parts - 1) ? (n - i * part_size) : part_size;
        res.emplace_back(pool.enqueue(assign_partition, arr, len, parts, n_parts));
    }

    for (auto& f: res) {
        f.get();
    }
    res.clear();


    for (size_t i = 0; i < n_parts; ++i) {
        res.emplace_back(pool.enqueue(fisher_yates, parts[i]));
    }
    
    for (auto& f: res) {
        f.get();
    }
    res.clear();

    auto end = std::chrono::high_resolution_clock::now();
    printf("Shuffle takes: %.6fs\n", std::chrono::duration<double>(end - begin).count());

    delete[] data;
    delete[] parts;

    return 0;
}
