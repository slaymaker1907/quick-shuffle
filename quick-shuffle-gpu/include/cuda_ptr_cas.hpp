#ifndef CUDA_PTR_CAS
#define CUDA_PTR_CAS

#include "cuda_polyfill.hpp"

namespace cuda_permute
{

template <class T>
T* cuda_ptr_cas(T **address, T *compare, T *value) {
    // all but one condition should be eliminated.
    if (sizeof(T*) == sizeof(unsigned long long int)) {
        unsigned long long int *cast_address = (unsigned long long int)address;
        unsigned long long int cast_compare = (unsigned long long int)compare;
        unsigned long long int case_value = (unsigned long long int)value;
        atomicCAS(cast_address, cast_compare, cast_value);
    } else {
        // assume it is the size of an int.
        unsigned int *cast_address = (unsigned int)address;
        unsigned int cast_compare = (unsigned int)compare;
        unsigned int case_value = (unsigned int)value;
        atomicCAS(cast_address, cast_compare, cast_value);
    }
}

}
#endif