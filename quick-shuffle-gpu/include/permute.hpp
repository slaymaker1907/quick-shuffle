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
#include "cub.cuh"
#include <chrono>

namespace cuda_permute
{

struct QuickPermuteConfig {
    size_t *partition_sizes;
    int pcount;
    int eles_per_thread;
    int shared_size;
    int seed;
    int bucket_size;
};

QuickPermuteConfig QuickPermuteConfig_new(int input_size, int seed) {
    QuickPermuteConfig result;

    result.pcount = 32;
    result.eles_per_thread = 64;
    result.seed = seed;
    result.bucket_size = 32;
    result.partition_sizes = nullptr;
    result.shared_size = 20 * 1024;

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

__device__ int bounded_rand(curandState_t *state, unsigned int max_val) {
    unsigned int rng_threshold = INT_MAX - (INT_MAX % max_val);
    return bounded_rand(state, max_val, rng_threshold);
}

template <typename T>
__global__ void quick_permute_partition(UnrolledList<T> *input,
                                        UnrolledList<T> *output,
                                        QuickPermuteConfig *config) {
    // Need to apply partitioning recursively.
    extern __shared__ char dyn_shared[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int *bucket_sizes = (int*)dyn_shared;
    T *bucket_data = (T*)(bucket_sizes + config->pcount);
    int bucket_data_size = config->pcount * sizeof(T);

    int *buffer_occupied = (int*)dyn_shared;
    T *input_buffer = (T*)(buffer_occupied + 1);
    int input_buffer_offset = sizeof(int) * config->pcount // bucket_sizes
                            + bucket_data_size             // bucket_data
                            + sizeof(int);                 // buffer_occupied (size used).
    int input_buffer_size = config->shared_size - input_buffer_offset;
    int input_buffer_eles = input_buffer_size / sizeof(T);
    int buffer_per_thread = input_buffer_eles / blockDim.x;
    int tile_iters_needed = divceil(config->eles_per_thread, buffer_per_thread);

    // now initialize bucket sizes.
    for (int partition = threadIdx.x; partition < config->pcount; partition += blockDim.x) {
        bucket_sizes[partition] = 0;
    }

    // sync to synchronized shared memory input buffer and the zeroed bucket sizes.
    __syncthreads();

    int rng_threshold = INT_MAX - (INT_MAX % config->pcount);

    curandState_t rand_state;
    curand_init(config->seed, tid, 0, &rand_state); // sequence should be tid to ensure independence.

    // check in inner loop if valid so we can syncthreads.
    for (int tile = 0; tile < tile_iters_needed; tile++) {
        // fill the buffer.
        if (threadIdx.x == 0) {
            // first copy over data from main memory to shared memory.
            int remaining = input_buffer_size / sizeof(T);
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

        //int tile_offset = tile * buffer_per_thread;

        // sync the buffer.
        __syncthreads();

        for (int i = 0; i < buffer_per_thread; i++) {
            int input_index = threadIdx.x + blockDim.x * i;
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

    QuickPermuteKernel() {
    }
    void run(cudaStream_t stream) {
        quick_permute_partition<<<block_count, block_size, shared_size, stream>>>(input, output, config);
    }
};

// template<typename T>
// class SizedArray {
// public:
//     size_t size;
//     T[] data;
//     __device__ void append(T* buffer, int buffer_size) {
//         size_t start = atomicAdd((unsigned long long*)size, (unsigned long long)buffer_size);
//         memcpy(&data[start], buffer, buffer_size * sizeof(T));
//     }
// }

template<typename T>
__global__ void concat_partitions(UnrolledList<T> *partitions, int pcount, T* output, size_t *cumm_part_sizes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    UnrolledList<T> input = tid < pcount ? partitions[tid] : partitions[0];
    int input_block_size = input.block_size;
    size_t start_ind = tid == 0 || tid >= pcount ? 0 : cumm_part_sizes[tid-1];
    T* thr_out = output + start_ind;

    __shared__ int max_node_count;
    if (threadIdx.x == 0) {
        max_node_count = 0;
    }
    __syncthreads();
    if (tid < pcount) {
        int my_count = input.node_count();
        atomicMax(&max_node_count, my_count);
    }
    __syncthreads();

    for (int i = 0; i < max_node_count; i++) {
        if (tid < pcount && !input.is_empty()) {
            thr_out += input.pop_to_buffer(thr_out, input_block_size);
        }
        // sync to avoid too much bandwidth.
        __syncthreads();
    }
}

template<typename T>
class ConcatPartitionsKernel {
public:
    UnrolledList<T> *partitions;
    T* output;
    size_t *cumm_part_sizes;
    int pcount;
    void run(cudaStream_t stream) {
        int block_size = 32;
        int block_count = divceil(pcount, block_size);
        concat_partitions<<<block_count, block_size>>>(partitions, pcount, output, cumm_part_sizes);
    }
};

__global__ void generate_radix_keys(int *keys, int size, int eles_per_thread, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ele_id =  eles_per_thread * blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t rand_state;
    curand_init(seed, tid, 0, &rand_state); // sequence should be tid to ensure independence.

    for (int i = 0; i < eles_per_thread; i++) {
        if (ele_id < size) {
            // this should work.
            keys[ele_id] = bounded_rand(&rand_state, ele_id);
        }
        __syncthreads();
    }
}

// input should be either in GPU memory or mapped memory.
template <typename T>
void quick_permute(T *input, size_t size) {
    // (1) partition the data into smaller chunks to avoid memory overhead of key per element
    // (2) compute the cummulative sum of the partition sizes
    // (3) collect the partitions back into a single array
    // (4) serially generate a key for each element and sort with radix sort.

    cudaDeviceProp device_properties;
    check_cuda_error(cudaGetDeviceProperties(&device_properties, 0));
    init_device_properties();
    auto config = QuickPermuteConfig_new(size, 8675309); // seed with a constant to have consistincy for now.

    // need to compute eles_per_thread based on data size.
    int max_blocks = 128;
    int threads_per_block = 32;
    int min_eles_per_thread = std::max(32 * sizeof(int) / sizeof(T), (size_t)1);
    int total_max_threads = max_blocks * threads_per_block;
    config.eles_per_thread = std::max((int)divceil(size, (size_t)total_max_threads), min_eles_per_thread);

    QuickPermuteConfig *device_config;
    // device_config->
    check_cuda_error(cudaMalloc((void**)&device_config, sizeof(QuickPermuteConfig)));
    check_cuda_error(cudaMemcpy(device_config, &config, sizeof(QuickPermuteConfig), cudaMemcpyHostToDevice));

    QuickPermuteKernel<T> default_kernel;
    default_kernel.config = device_config;

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

    default_kernel.block_size = threads_per_block;
    default_kernel.block_count = std::min(max_blocks, CCAST(divceil(size, (size_t)threads_per_block * config.eles_per_thread), int));

    default_kernel.shared_size = config.shared_size;
    // std::cout << "Shared size:" << default_kernel.shared_size << std::endl;
    // std::cout << "Shared size:" << device_properties.sharedMemPerBlockOptin << std::endl;
    // std::cout << "Shared size:" << device_properties.sharedMemPerMultiprocessor << std::endl;

    size_t *dummy_size_t = nullptr;
    size_t *host_sizes = alloc_pinned_array(config.pcount, dummy_size_t);
    memset(host_sizes, 0, sizeof(size_t) * config.pcount);
    size_t *device_sizes;
    check_cuda_error(cudaMalloc((void**)&device_sizes, sizeof(size_t) * config.pcount));

    default_kernel.input = device_input;
    default_kernel.output = device_output;

    auto start_chron = std::chrono::high_resolution_clock::now();
    default_stream.memcpy_to_device(device_sizes, host_sizes, config.pcount);
    CudaEvent *start = default_stream.create_event();
    printf("Starting to launch kernel.\n");
    default_stream.launch_kernel(default_kernel);
    CudaEvent *end = default_stream.create_event();
    default_stream.memcpy_to_host(host_sizes, device_sizes, config.pcount);
    default_stream.join();

    printf("Took %fms to execute partition kernel.\n", CudaEvent_elapsed_time(start, end));
    delete start;
    delete end;

    size_t last_size_value = 0;
    int max_size_value = 0;
    for(int i = 0; i < config.pcount; i++) {
        size_t temp = host_sizes[i];
        host_sizes[i] += last_size_value;
        last_size_value = temp;
        max_size_value = (int)std::max(last_size_value, (size_t)max_size_value);
    }

    T *device_final_output;
    check_cuda_error(cudaMalloc((void**)&device_final_output, sizeof(T) * size));
    start = default_stream.create_event();
    default_stream.memcpy_to_device(device_sizes, host_sizes, config.pcount);

    ConcatPartitionsKernel<T> concat_kernel;
    concat_kernel.partitions = device_output;
    concat_kernel.pcount = config.pcount;
    concat_kernel.cumm_part_sizes = device_sizes;
    concat_kernel.output = device_final_output;

    default_stream.launch_kernel(concat_kernel);
    end = default_stream.create_event();
    default_stream.join();

    printf("Took %fms to execute collect partitions.\n", CudaEvent_elapsed_time(start, end));
    delete start;
    delete end;

    start = default_stream.create_event();

    int *device_keys;
    check_cuda_error(cudaMalloc((void**)&device_keys, sizeof(int) * max_size_value));

    int *device_keys_out;
    check_cuda_error(cudaMalloc((void**)&device_keys_out, sizeof(int) * max_size_value));

    for (int partition = 0; partition < config.pcount; partition++) {
        int partition_size;
        T *values_input;
        T *values_output;
        if (partition == 0) {
            partition_size = host_sizes[partition];
            values_input = device_final_output;
            values_output = input;
        } else {
            partition_size = host_sizes[partition] - host_sizes[partition-1];
            values_input = device_final_output + host_sizes[partition-1];
            values_output = input + host_sizes[partition-1];
        }
        int keys_block_count = divceil(partition_size, 32 * config.eles_per_thread);
        generate_radix_keys<<<keys_block_count, 32>>>(device_keys, partition_size, config.eles_per_thread, 8675309 + 1 + partition);
        check_cuda_error(cudaDeviceSynchronize());

        void *temp_storage = nullptr;
        // first is to determine storage requirements.
        size_t temp_storage_size = 0;
        cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_size, device_keys,
                device_keys_out, values_input, values_output, partition_size, 0, sizeof(int) * 8, default_stream.wrapped_stream);

        check_cuda_error(cudaMalloc(&temp_storage, temp_storage_size));
        cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_size, device_keys,
                device_keys_out, values_input, values_output, partition_size, 0, sizeof(int) * 8, default_stream.wrapped_stream);
        default_stream.join();
        cudaFree(temp_storage);
    }

    end = default_stream.create_event();
    default_stream.join();
    printf("Took %fms to sort all partitions.\n", CudaEvent_elapsed_time(start, end));
    delete start;
    delete end;

    auto end_chron = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(end_chron - start_chron);
    std::cout << "Elapsed time: " << elapsed.count() * 1000.0 << " ms" << std::endl;

    // std::vector<UnrolledList<T>*> second_device_output(0);
    // second_device_output.reserve(config.pcount);

    // std::vector<CudaStream*> partition_streams;
    // partition_streams.reserve(config.pcount);

    // for (int i = 0; i < config.pcount; i++) {
    //     auto stream = new CudaStream();
    //     partition_streams.push_back(stream);
    //     UnrolledList<T> *device_temp;
    //     check_cuda_error(cudaMalloc((void**)&device_temp, output_list_size));
    //     second_device_output.push_back(device_temp);
    //     stream->memcpy_to_device(device_temp, host_output, config.pcount);
    // }

    // // wait for default stream before continuing since we need the size of each partition to determine # of blocks.
    // default_stream.join();

    // // TODO: make these global calls asynchronously and limit block_count.
    // for (int i = 0; i < config.pcount; i++) {
    //     CudaStream *stream = partition_streams[i];
    //     default_kernel.new_block_count(host_sizes[i]);
    //     default_kernel.input = &device_output[i];
    //     default_kernel.output = second_device_output[i];
    //     stream->launch_kernel(default_kernel);
    // }

    // check_cuda_error(cudaDeviceSynchronize());

    // for (CudaStream *stream : partition_streams) {
    //     delete stream;
    // }

    free_pinned(host_sizes);
    free_pinned(host_output);
}

}



#endif
