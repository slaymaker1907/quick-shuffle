#include "permute.hpp"

namespace cuda_permute {

QuickPermuteConfig QuickPermuteConfig_new(int input_size, int seed) {
    QuickPermuteConfig result;
    result.pcount = 64;
    result.eles_per_thread = 8;
    result.seed = seed;
    result.bucket_size = 8;
    result.global_bucket_size = 2 * input_size / result.pcount;
}

int QuickPermuteConfig_global_memory_size(QuickPermuteConfig *config) {
    return (config->pcount + config->global_bucket_size * config->pcount) * sizeof(int);
}

__device__ int bounded_rand(curandState_t *state, unsigned int max_val, unsigned int threshold) {
    // Adapted from https://stackoverflow.com/questions/51344558/uniform-random-numbers-dont-include-upper-bound
    unsigned int current;
    do {
        current = curand(state);
    } while (current >= threshold);

    return current % max_val;
}

__global__ void quick_permute_partition(int *const input, int *output, int size, QuickPermuteConfig *const config) {
    // Need to apply partitioning recursively.
    extern __shared__ volatile int *dyn_shared;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ele_id = blockIdx.x * blockDim.x * config->eles_per_thread + threadIdx.x;
    int last_ele = min(blockDim.x * config->eles_per_thread * (blockIdx.x + 1), size);
    unsigned int pcount = config->pcount;
    int i;
    int rng_threshold = UINT_MAX - (UINT_MAX % pcount);

    curandState_t rand_state;
    curand_init(config->seed, tid, 0, &rand_state); // sequence should be tid to ensure independence.
    volatile int *bucket_sizes = dyn_shared;
    int bucket_size_size = config->pcount * sizeof(int);
    volatile int *bucket_data = dyn_shared + bucket_size_size;
    int *global_bucket_sizes = output;
    int *global_bucket_data = output + bucket_size_size;

    if (threadIdx.x == 0) {
        for (i = 0; i < pcount; i++) {
            bucket_sizes[i] = 0;
        }
    }
    __syncthreads();

    while (ele_id < last_ele) {
        int partition = bounded_rand(&rand_state, pcount, rng_threshold);
        volatile int *pbucket_size = bucket_sizes + partition;
        volatile int *pbucket_data = bucket_data + partition * config->bucket_size;
INCREMENT_SIZE:
        int new_size = atomicAdd((int*)pbucket_size, 1) + 1;

        // make sure to empty out bucket when full to avoid deadlock and allow easy busy wait.
        if (new_size >= config->bucket_size) {
            // Maybe introduce another layer of cache in global memory, non-reduced.
            // then do a combining step for these groups.
            if (new_size == config->bucket_size) {
                int old_glob_size = atomicAdd(global_bucket_sizes + partition, config->bucket_size);
                assert(old_glob_size < config->global_bucket_size);
                int *data_start = global_bucket_data + partition * config->global_bucket_size + old_glob_size;
                for (i = 0; i < config->bucket_size; i++) {
                    data_start[i] = pbucket_data[i];
                }
                __threadfence_block(); // cannot allow other threads to write to data before we have written shared to global.
                *pbucket_size = 0;
            }

            // hopefully nvcc can figure out the scheduling for this.
            // not our responsibility to empty even if ==.
            while (*pbucket_size >= config->bucket_size) {
                // busy wait.
            }
            goto INCREMENT_SIZE;
        }
    }

    __syncthreads();
    if (threadIdx.x < config->bucket_size) {
        // don't need volatile anymore since we ran __syncthreads().
        int bucket_size = ((int*)bucket_sizes)[threadIdx.x];
        int old_glob_size = atomicAdd(global_bucket_sizes + threadIdx.x, bucket_size);
        int *shared_data_start = (int*)bucket_data + threadIdx.x * config->bucket_size;
        int *glob_data_start = global_bucket_data + threadIdx.x * config->global_bucket_size + old_glob_size;
        for (i = 0; i < bucket_size; i++) {
            glob_data_start[i] = shared_data_start[i];
        }
    }
}

void quick_permute_helper(int *input, int size) {
    auto config = QuickPermuteConfig_new(size, 8675309); // seed with a constant to have consistincy for now.
    QuickPermuteConfig *device_config;
    cudaError_t config_alloc_err = cudaMalloc((void**)&device_config, sizeof(QuickPermuteConfig));
    cudaError_t copy_config_err = cudaMemcpy(device_config, &config, sizeof(QuickPermuteConfig), cudaMemcpyHostToDevice);
    int *permute_output;
    int *permute_input;

    cudaError_t alloc_err = cudaMalloc((void**)&permute_output, QuickPermuteConfig_global_memory_size(&config));
    cudaError_t input_alloc_err = cudaMalloc((void**)&permute_input, sizeof(int) * size);
    cudaError_t input_copy_err = cudaMemcpy(permute_input, input, size * sizeof(int), cudaMemcpyHostToDevice);
    const int block_size = 32;
    int eles_per_block = config.eles_per_thread * block_size;
    int block_count = size / eles_per_block;
    if (size % eles_per_block != 0) {
        ++block_count;
    }

    int shared_size = (config.pcount + config.bucket_size * config.pcount) * sizeof(int);

    // quick_permute_partition<<<block_count, block_size, shared_size>>>(input, permute_output, size, device_config);
}

}
