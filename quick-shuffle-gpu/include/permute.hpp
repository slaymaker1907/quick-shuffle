#ifndef CUDA_PERMUTE
#define CUDA_PERMUTE

#include "cuda_polyfill.hpp"
#include <limits.h>
#include <vector>
#include <assert.h>
#include <climits>
#include <cmath>
#include "unrolled_list.hpp"

namespace cuda_permute
{

struct QuickPermuteConfig {
    int pcount;
    int eles_per_thread;
    int seed;
    int bucket_size;
    int global_bucket_size;
};

QuickPermuteConfig QuickPermuteConfig_new(int input_size, int seed);

int QuickPermuteConfig_global_memory_size(QuickPermuteConfig *config);

__device__ int bounded_rand(curandState_t *state, unsigned int max_val, unsigned int threshold);
// output and global_bucket_size should be large enough s.t. we rarely overflow a bucket (will fail if violated).
__global__ void quick_permute_partition(UnrolledList<int> *input,
                                        UnrolledList<int> *output,
                                        int size,
                                        QuickPermuteConfig *const config);
void quick_permute_helper(int *input, int size);

template <typename T>
void quick_permute(std::vector<T> data) {
    // TODO: add error handling.
    static_assert(sizeof(T) == sizeof(int));
    assert(data.size() < (size_t)INT_MAX);
    T *data_ptr = data.data();
    quick_permute_helper((int*)data_ptr, data.size());
}

// class IRng {
// public:
//     int gen_int(int bound);
//     IRng();
// };


}



#endif