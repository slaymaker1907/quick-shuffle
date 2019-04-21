#ifndef CUDA_PERMUTE
#define CUDA_PERMUTE

#include "cuda_polyfill.hpp"
#include <limits.h>
#include <vector>
#include <assert.h>
#include <climits>
#include <cmath>
#include "permute_util.hpp"
#include "unrolled_list.hpp"
#include "stream_pool.hpp"

namespace cuda_permute
{

struct QuickPermuteConfig {
    size_t *partition_sizes;
    int pcount;
    int eles_per_thread;
    int seed;
    int bucket_size;
};

QuickPermuteConfig QuickPermuteConfig_new(int input_size, int seed) {
    QuickPermuteConfig result;

    result.pcount = 32;
    result.eles_per_thread = 8;
    result.seed = seed;
    result.bucket_size = 32;
    result.partition_sizes = nullptr;

    return result;
}

__device__ int bounded_rand(curandState_t *state, unsigned int max_val, unsigned int threshold) {
    // Adapted from https://stackoverflow.com/questions/51344558/uniform-random-numbers-dont-include-upper-bound
    unsigned int current;
    do {
        current = curand(state);
    } while (current >= threshold);

    return current % max_val;
}

template <typename T>
__global__ void quick_permute_partition(UnrolledList<T> *input,
                                        UnrolledList<T> *output,
                                        QuickPermuteConfig *config) {
    // Need to apply partitioning recursively.
    extern __shared__ char dyn_shared[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int *buffer_occupied = (int*)dyn_shared;
    T *input_buffer = (T*)(buffer_occupied + 1);
    int input_buffer_size = config->eles_per_thread * blockDim.x * sizeof(T);
    int *bucket_sizes = (int*)(input_buffer + input_buffer_size);
    T *bucket_data = (T*)(bucket_sizes + config->pcount);
    int bucket_data_size = config->pcount * sizeof(T);

    if (threadIdx.x == 0) {
        // first copy over data from main memory to shared memory.
        int remaining = config->eles_per_thread * blockDim.x;
        *buffer_occupied = 0;
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
        size_t input_index = (size_t)threadIdx.x + blockDim.x * i;
        int partition = bounded_rand(&rand_state, config->pcount, rng_threshold);

        // first move to partition if we were able to fit it in the partition.
        int *pbucket_size = bucket_sizes + partition;
        T *pbucket_data = bucket_data + bucket_data_size;

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
                if (config->partition_sizes) {
                    atomicAdd((unsigned long long int*)&config->partition_sizes[partition], (unsigned long long int)config->bucket_size);
                }
                output[partition].push_many(pbucket_data, config->bucket_size);
                *pbucket_size = 0;
            }
        } while (__syncthreads_or(move_element));
    }

    // don't need __syncthreads since the above loop always ends with one.
    // __syncthreads();

    // now copy any remaining elements back to global memory.
    // not sure if one thread might be better. better if we have global collisions?
    if (threadIdx.x == 0) {
        for (int partition = 0; partition < config->pcount; partition++) {
            if (partition < config->pcount && bucket_sizes[partition] > 0) {
                output[partition].push_many(bucket_data + partition * config->bucket_size, bucket_sizes[partition]);
            }
        }
    }
}

template <typename T>
class QuickPermuteKernel {
public:
    size_t shared_size;
    UnrolledList<T> *input, *output;
    QuickPermuteConfig *config;
    int block_count;
    int block_size;
    int eles_per_thread;

    QuickPermuteKernel() {
    }
    void run(cudaStream_t stream) {
        quick_permute_partition<<<block_count, block_size, shared_size, stream>>>(input, output, config);
    }
    void new_block_count(size_t input_size) {
        int eles_per_block = eles_per_thread * block_size;
        block_count = (int)(input_size / (size_t)eles_per_block);
        if (((size_t) block_count) * ((size_t) eles_per_block) < input_size) {
            block_count++;
        }
    }
};

template <typename T>
void quick_permute(std::vector<T> data) {
    T *input = data.data();
    size_t size = data.size();

    cudaDeviceProp device_properties;
    check_cuda_error(cudaGetDeviceProperties(&device_properties, 0));
    auto config = QuickPermuteConfig_new(size, 8675309); // seed with a constant to have consistincy for now.
    QuickPermuteConfig *device_config;
    check_cuda_error(cudaMalloc((void**)&device_config, sizeof(QuickPermuteConfig)));
    check_cuda_error(cudaMemcpy(device_config, &config, sizeof(QuickPermuteConfig), cudaMemcpyHostToDevice));

    QuickPermuteKernel<T> default_kernel;
    default_kernel.config = device_config;
    default_kernel.eles_per_thread = config.eles_per_thread;

    UnrolledList<T> *device_input = move_to_gpu(input, size);
    UnrolledList<T> *device_output;
    size_t output_list_size = sizeof(UnrolledList<T>) * config.pcount;
    check_cuda_error(cudaMalloc((void**)&device_output, output_list_size));
    UnrolledList<T> *dummy_list = nullptr;
    UnrolledList<T> *host_output = alloc_pinned_array(config.pcount, dummy_list);
    assert(host_output);

    UnrolledList<T> default_output(512); // arbitrarily chose size to be 512.

    for (int i = 0; i < config.pcount; i++) {
        host_output[i] = default_output;
    }

    // make sure input fully copied before we start copying data other data.
    check_cuda_error(cudaDeviceSynchronize());

    CudaStream default_stream;
    default_stream.memcpy_to_device(device_output, host_output, config.pcount);

    default_kernel.block_size = 32;
    default_kernel.new_block_count(size);

    default_kernel.shared_size = device_properties.sharedMemPerBlock;
    std::cout << "Shared size:" << default_kernel.shared_size << std::endl;
    std::cout << "Shared size:" << device_properties.sharedMemPerBlockOptin << std::endl;
    std::cout << "Shared size:" << device_properties.sharedMemPerMultiprocessor << std::endl;

    size_t *dummy_size_t = nullptr;
    size_t *host_sizes = alloc_pinned_array(config.pcount, dummy_size_t);
    memset(host_sizes, 0, sizeof(size_t) * config.pcount);
    size_t *device_sizes;
    check_cuda_error(cudaMalloc((void**)&device_sizes, sizeof(size_t) * config.pcount));

    default_kernel.input = device_input;
    default_kernel.output = device_output;

    default_stream.memcpy_to_device(device_sizes, host_sizes, config.pcount);
    default_stream.launch_kernel(default_kernel);
    check_cuda_error(cudaDeviceSynchronize());
    default_stream.memcpy_to_host(host_sizes, device_sizes, config.pcount);

    std::vector<UnrolledList<T>*> second_device_output(0);
    second_device_output.reserve(config.pcount);

    std::vector<CudaStream*> partition_streams;
    partition_streams.reserve(config.pcount);

    for (int i = 0; i < config.pcount; i++) {
        auto stream = new CudaStream();
        partition_streams.push_back(stream);
        UnrolledList<T> *device_temp;
        check_cuda_error(cudaMalloc((void**)&device_temp, output_list_size));
        second_device_output.push_back(device_temp);
        stream->memcpy_to_device(device_temp, host_output, config.pcount);
    }

    // wait for default stream before continuing since we need the size of each partition to determine # of blocks.
    default_stream.join();

    // TODO: make these global calls asynchronously and limit block_count.
    for (int i = 0; i < config.pcount; i++) {
        CudaStream *stream = partition_streams[i];
        default_kernel.new_block_count(host_sizes[i]);
        default_kernel.input = &device_output[i];
        default_kernel.output = second_device_output[i];
        stream->launch_kernel(default_kernel);
    }

    check_cuda_error(cudaDeviceSynchronize());

    for (CudaStream *stream : partition_streams) {
        delete stream;
    }

    free_pinned(host_output);
}

}



#endif