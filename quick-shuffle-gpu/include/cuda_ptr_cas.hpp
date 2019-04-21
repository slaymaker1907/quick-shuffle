#ifndef CUDA_PTR_CAS
#define CUDA_PTR_CAS

#include "cuda_polyfill.hpp"

namespace cuda_permute
{

template <class T>
__device__ T* cuda_ptr_cas(T **address, T *compare, T *value) {
    static_assert(sizeof(T*) == sizeof(unsigned long long int));
    unsigned long long int *cast_address = (unsigned long long int*)address;
    unsigned long long int cast_compare = (unsigned long long int)compare;
    unsigned long long int cast_value = (unsigned long long int)value;
    return (T*)atomicCAS(cast_address, cast_compare, cast_value);
}

template <class T>
__device__ T* cuda_ptr_cas(volatile T **address, T *compare, T *value) {
    return cuda_ptr_cas((T**)address, compare, value);
}

}
#endif