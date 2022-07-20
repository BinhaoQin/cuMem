#ifndef __cuManagedAllocator_cuh__
#define __cuManagedAllocator_cuh__

#include "cuBase.h"
#include <new>

template <typename T> class ManagedAllocator {
public:
  static DEVICE_INDEPENDENT T *allocate(std::size_t n = 1) {
    T *result = nullptr;
    cuAssert(cudaMallocManaged(&result, n * sizeof(T)));
    return result;
  }

  static inline DEVICE_INDEPENDENT void deallocate(T *p) {
    cuAssert(cudaFree(p));
  }

  template <typename... Args>
  static DEVICE_INDEPENDENT T *construct(Args &&...args) {
    T *result = allocate();
    new (result) T(args...);
    return result;
  }

  static DEVICE_INDEPENDENT void destruct(T *p) {
    p->~T();
    deallocate(p);
  }
};

#endif
