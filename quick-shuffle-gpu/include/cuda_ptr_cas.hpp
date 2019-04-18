#ifndef CUDA_PTR_CAS
#define CUDA_PTR_CAS

#include <cuda_ptr_cas.hpp>

namespace cuda_permute
{

template <class T>
T* cuda_ptr_cas(T **address, T *compare, T *value);

}
#endif