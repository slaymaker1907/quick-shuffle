#ifndef PERMUTE_UTIL
#define PERMUTE_UTIL

// taken from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define check_cuda_error(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while(0)

#define CCAST(expr, type) ((type)(expr))

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#include "cuda_polyfill.hpp"
#include "stream_pool.hpp"
#include <string>
#include <assert.h>
#include <algorithm>

namespace cuda_permute
{

cudaDeviceProp DEVICE_PROPERTIES;

void init_device_properties() {
   check_cuda_error(cudaGetDeviceProperties(&DEVICE_PROPERTIES, 0));
}

size_t max_cuda_threads() {
   return DEVICE_PROPERTIES.maxThreadsPerMultiProcessor + DEVICE_PROPERTIES.multiProcessorCount;
}

template<typename NumberT>
__device__ __host__ NumberT divceil(NumberT a, NumberT b) {
   NumberT result = a / b;
   result += (result * b < a) ? 1 : 0;
   return result;
}

template<typename NumberT>
NumberT round_to_multiple(NumberT x, NumberT multiple_of) {
   return divceil(x, multiple_of) * multiple_of;
}

__global__ void malloc_on_gpu_kernel(void** result, size_t alloc_size) {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   result[tid] = malloc(alloc_size);
   assert(result[tid]);
}

class MallocOnGpuKernel {
public:
   void** result;
   size_t alloc_size;
   int alloc_count;
   void run(cudaStream_t stream) {
      malloc_on_gpu_kernel<<<alloc_count, 1, 0, stream>>>(result, alloc_size);
   }
};

// T** should be alloc_count long and be allocated in page-locked memory.
template<typename T>
void malloc_on_gpu(T** result, int alloc_count, CudaStream *stream, size_t alloc_size = 0) {
   assert(alloc_count > 0);
   if (alloc_size == 0) {
      alloc_size = sizeof(T);
   }

   void **gpu_result;
   check_cuda_error(cudaMalloc((void**)&gpu_result, sizeof(void*) * alloc_count));

   MallocOnGpuKernel kernel;
   kernel.result = gpu_result;
   kernel.alloc_size = alloc_size;
   kernel.alloc_count = alloc_count;

   stream->launch_kernel(kernel);

   stream->memcpy_to_host(result, (T**)gpu_result, alloc_count);
}

const int CACHE_LINE_SIZE = 32;

// move one T* per block.
__global__ void cudaMalloc_to_malloc_kernel(char* __restrict__ cudaMalloc_ptr,
                                            char* __restrict__ malloc_ptr,
                                            int mem_per_thread,
                                            size_t size) {
   size_t tid = mem_per_thread * blockIdx.x * blockDim.x + threadIdx.x;
   char* __restrict__ src_ptr = cudaMalloc_ptr + tid;
   char* __restrict__ dest_ptr = malloc_ptr + tid;

   int max_i;
   if (tid + CCAST(mem_per_thread, size_t) <= size) {
      max_i = mem_per_thread;
   } else if (tid < size) {
      max_i = CCAST(size - tid, int);
   } else {
      max_i = 0;
   }

   for (int i = 0; i < mem_per_thread; i++) {
      if (i < max_i) {
         dest_ptr[i] = src_ptr[i];
      }
      __syncthreads();
   }
}

class MallocToMallocKernel {
public:
   char* cudaMalloc_ptr;
   char* malloc_ptr;
   size_t size;

   void run(cudaStream_t stream) {
      size_t thread_target = max_cuda_threads();
      int mem_per_thread;
      if (thread_target * CCAST(CACHE_LINE_SIZE, size_t) < thread_target) {
         thread_target = divceil(size, CCAST(CACHE_LINE_SIZE, size_t));
         mem_per_thread = CACHE_LINE_SIZE;
      } else {
         mem_per_thread = (int)round_to_multiple(size, (size_t)CACHE_LINE_SIZE);
      }

      int block_size = 32; // arbitrary.
      int block_count = (int)divceil(size, CCAST(block_size, size_t) * CCAST(mem_per_thread, size_t));
      cudaMalloc_to_malloc_kernel<<<block_count, block_size, 0, stream>>>(cudaMalloc_ptr, malloc_ptr, mem_per_thread, size);
   }
};

template<typename T>
void cudaMalloc_to_malloc(T* malloc_ptr, T* cudaMalloc_ptr, size_t alloc_count, CudaStream *stream, size_t alloc_size = 0) {
   if (alloc_size == 0) {
      alloc_size = sizeof(T);
   }

   size_t size_to_move = alloc_size * alloc_count;
   MallocToMallocKernel kernel;
   kernel.cudaMalloc_ptr = (char*)cudaMalloc_ptr;
   kernel.malloc_ptr = (char*)malloc_ptr;
   kernel.size = size_to_move;

   stream->launch_kernel(kernel);
}

}

#endif
