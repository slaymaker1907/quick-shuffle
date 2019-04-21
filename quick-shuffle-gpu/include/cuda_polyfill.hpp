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
int atomicSub(int* address, int val);
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

void __threadfence();

struct cudaStream_t {
};
cudaError_t cudaStreamCreate(cudaStream_t *stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaMemcpyAsync(void *dest, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);

int __syncthreads_and(int predicate);
int __syncthreads_count(int predicate);
int __syncthreads_or(int predicate);

struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    size_t totalConstMem;
    int major;
    int minor;
    int clockRate;
    size_t textureAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int tccDriver;
};

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);

#define __syncthreads() do {} while(0)
#endif
#else
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>

// for some reason, blockIdx isn't showing up with VSCode.
// #ifdef VSCODE
// extern dim3 blockIdx;
// #endif

#endif
