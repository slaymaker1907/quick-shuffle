#include <iostream>
#include <vector>
#include "permute.hpp"
#include "cuda_polyfill.hpp"
#include "permute_util.hpp"

int* make_sized_vector(size_t size) {

    int *result;
    check_cuda_error(cudaMallocManaged((void**)&result, size * sizeof(int)));
    // result = cuda_permute::alloc_pinned_array(size, result);

    for (size_t i = 0; i < size; i++) {
        result[i] = i;
    }
    return result;
}

int main() {
    size_t shuffle_size = 100 * 1000 * 1000;
    check_cuda_error(cudaThreadSetLimit(cudaLimitMallocHeapSize, ((size_t)5) * 1000 * 1000 * 1000));
    auto to_shuffle = make_sized_vector(shuffle_size);
    std::cout << "Starting to shuffle." << std::endl;
    try {
        cuda_permute::quick_permute(to_shuffle, shuffle_size);
    } catch (char const* msg) {
        std::cout << "Shuffle was not successful." << std::endl;
        std::cout << msg << std::endl;
    }
    return 0;
}
