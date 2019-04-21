#include <iostream>
#include <vector>
#include "permute.hpp"
#include "cuda_polyfill.hpp"
#include "permute_util.hpp"

std::vector<int> make_sized_vector(size_t size) {
    std::vector<int> result(size);
    for (size_t i = 0; i < size; i++) {
        result[i] = i;
    }
    return result;
}

int main() {
    auto to_shuffle = make_sized_vector(10 * 1000 * 1000);
    check_cuda_error(cudaThreadSetLimit(cudaLimitMallocHeapSize, ((size_t)5) * 1000 * 1000 * 1000));
    std::cout << "Starting to shuffle." << std::endl;
    try {
        cuda_permute::quick_permute(to_shuffle);
    } catch (char const* msg) {
        std::cout << "Shuffle was not successful." << std::endl;
        std::cout << msg << std::endl;
    }
    return 0;
}
