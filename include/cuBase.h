#ifndef __cuBase_cuh__
#define __cuBase_cuh__

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#ifndef DEVICE_INDEPENDENT
#define DEVICE_INDEPENDENT __host__ __device__
#endif

DEVICE_INDEPENDENT inline cudaError_t cuErrorCheck(cudaError_t error_code,
                                                   const char *file, int line) {
  if (error_code != cudaSuccess) {
    fprintf(stderr, "CUDA Assertion Failed: %s at %s:%d\n",
            cudaGetErrorString(error_code), file, line);
  }
  return error_code;
}

#ifndef NDEBUG
#define cuAssert(error_code) cuErrorCheck((error_code), __FILE__, __LINE__)
#else
#define cuAssert(error_code) (error_code)
#endif

#endif
