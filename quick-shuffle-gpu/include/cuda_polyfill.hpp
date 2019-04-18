#ifndef CUDA_POLYFILL
#define CUDA_POLYFILL
#ifndef __NVCC__

// template <int a, int b>
// int func() {
// }
#define __global__
#define __device__
#define __shared__
#define __host__


struct CudaId {
    int x, y, z;
};
struct cudaError_t {
};

CudaId threadIdx;
CudaId blockIdx;
CudaId blockDim;

struct curandState_t {
};

unsigned int curand(curandState_t *state);
void curand_init (unsigned long long seed, unsigned long long sequence,
    unsigned long long offset, curandState_t *state);

int min(int a, int b);
int atomicAdd(int* address, int val);
void __threadfence_block();

cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cduaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};
cudaError_t cudaMemcpy(void *dest, const void *src, size_t count, cudaMemcpyKind kind);

int atomicCAS(int *address, int compare, int val);
unsigned long long int atomicCAS(unsigned long long int *address, unsigned long long int compare, unsigned long long int val);

#define __syncthreads() do {} while(0)
#endif
#endif
