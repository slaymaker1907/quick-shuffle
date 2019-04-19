#include "permute.hpp"
#include "unrolled_list.hpp"

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

#define T char
__global__ void quick_permute_partition(UnrolledList<T> *input,
                                        UnrolledList<T> *output,
                                        int size,
                                        QuickPermuteConfig const* config) {
    // Need to apply partitioning recursively.
    extern __shared__ void *dyn_shared;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int *buffer_occupied = (int*)dyn_shared;
    T *input_buffer = (T*)(buffer_occupied + 1);
    int input_buffer_size = config->eles_per_thread * blockDim.x;
    int *bucket_sizes = (int*)(input_buffer + input_buffer_size);
    T *bucket_data = (T*)(bucket_sizes + config->pcount);
    int bucket_data_size = config->pcount * sizeof(T);

    if (threadIdx.x == 0) {
        // first copy over data from main memory to shared memory.
        int remaining;
        do {
            int filledc = input->pop_to_buffer(input_buffer + (*buffer_occupied), remaining);
            *buffer_occupied = (*buffer_occupied) + filledc;
            if (filledc == 0) {
                // no more elements in input.
                break;
            }

            remaining = input_buffer_size - (*buffer_occupied);
        } while (remaining > 0);
    }
    // now initialize bucket sizes.
    for (int partition = threadIdx.x; partition < config->pcount; partition += blockDim.x) {
        bucket_sizes[partition] = 0;
    }

    // sync to synchronized shared memory input buffer and the zeroed bucket sizes.
    __syncthreads();

    int rng_threshold = UINT_MAX - (UINT_MAX % config->pcount);

    curandState_t rand_state;
    curand_init(config->seed, tid, 0, &rand_state); // sequence should be tid to ensure independence.

    // check in inner loop if valid so we can syncthreads.
    for (int i = 0; i < config->eles_per_thread; i++) {
        int input_index = threadIdx.x + blockDim.x * i;
        int partition = bounded_rand(&rand_state, config->pcount, rng_threshold);

        // first move to partition if we were able to fit it in the partition.
        int *pbucket_size = bucket_sizes + partition;
        T *pbucket_data = bucket_data + partition * config->bucket_size;

        int old_size = 0;
        bool move_element = input_index < (*buffer_occupied);

        do {
            // first increment size of the partition.
            if (move_element) {
                old_size = atomicAdd(pbucket_size, 1);
            }

            // check to see if we can write to the bucket.
            if (move_element && (old_size < config->bucket_size)) {
                pbucket_data[old_size] = input_buffer[input_index];
                move_element = false;
            }

            // sync in case we need to move to global memory.
            __syncthreads();

            // make more room if full.
            if (move_element && (old_size == config->bucket_size)) {
                output[partition].push_many(pbucket_data, config->bucket_size);
                *pbucket_size = 0;
            }
        } while (__syncthreads_or(move_element));
    }

    // don't need __syncthreads since the above loop always ends with one.
    // __syncthreads();

    // now copy any remaining elements back to global memory.
    // not sure if one thread might be better. better if we have global collisions?
    for (int partition = threadIdx.x; partition < config->pcount; partition += blockDim.x) {
        if (partition < config->pcount && bucket_sizes[partition] > 0) {
            output[partition].push_many(bucket_data + partition * config->bucket_size, bucket_sizes[partition]);
        }
    }
}
#undef T

#define T int
void quick_permute_helper(T const* input, int size) {
    cudaDeviceProp device_properties;
    cudaError_t query_err = cudaGetDeviceProperties(&device_properties, 0);
    auto config = QuickPermuteConfig_new(size, 8675309); // seed with a constant to have consistincy for now.
    QuickPermuteConfig *device_config;
    cudaError_t config_alloc_err = cudaMalloc((void**)&device_config, sizeof(QuickPermuteConfig));
    cudaError_t copy_config_err = cudaMemcpy(device_config, &config, sizeof(QuickPermuteConfig), cudaMemcpyHostToDevice);


    UnrolledList<T> *device_input = move_to_gpu(input, size);
    UnrolledList<T> *device_output;
    size_t output_list_size = sizeof(UnrolledList<T>) * config.pcount;
    cudaError_t device_output_alloc_err = cudaMalloc((void**)&device_output, output_list_size);
    UnrolledList<T> *host_output = (UnrolledList<T>*)malloc(output_list_size);
    assert(host_output);

    UnrolledList<T> default_output(512);

    for (int i = 0; i < config.pcount; i++) {
        host_output[i] = default_output;
    }

    cudaError_t output_init_copy_err = cudaMemcpy(device_output, host_output, output_list_size, cudaMemcpyHostToDevice);

    const int block_size = 32;
    int eles_per_block = config.eles_per_thread * block_size;
    int block_count = size / eles_per_block;
    if (size % eles_per_block != 0) {
        ++block_count;
    }

    size_t approx_stack_needed = 0;
    size_t shared_size = device_properties.sharedMemPerBlock - approx_stack_needed;

    // quick_permute_partition<<<block_count, block_size, shared_size>>>(device_input, device_output, size, device_config);
}
#undef T

}
