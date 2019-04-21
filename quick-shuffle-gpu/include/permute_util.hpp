#ifndef PERMUTE_UTIL
#define PERMUTE_UTIL

#include "cuda_polyfill.hpp"
#include <string>

// namespace cuda_permute
// {

// taken from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define check_cuda_error(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while(0)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// }

#endif
