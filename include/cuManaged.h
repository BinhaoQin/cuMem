#ifndef __cuManaged_cuh__
#define __cuManaged_cuh__

#include "cuManagedAllocator.h"

template <typename T, bool use_new = true> class Managed {
protected:
  T *ptr;
  ManagedAllocator<T> allocator;

public:
  template <typename... Args>
  DEVICE_INDEPENDENT Managed(Args &&...args) : ptr(nullptr), allocator() {
    if constexpr (use_new)
      ptr = allocator.construct(args...);
    else
      ptr = allocator.allocate();
  }

  DEVICE_INDEPENDENT ~Managed() {
    if constexpr (use_new)
      allocator.destruct(ptr);
    else
      allocator.deallocate(ptr);
    ptr = nullptr;
  }

  template <bool n>
  DEVICE_INDEPENDENT Managed(const Managed<T, n> &source) = delete;

  DEVICE_INDEPENDENT T *operator->() const { return ptr; }

  cudaError_t prefetch(int device = cudaCpuDeviceId,
                       cudaStream_t stream = nullptr) const {
    return cuAssert(cudaMemPrefetchAsync(ptr, sizeof(T), device, stream));
  }

  DEVICE_INDEPENDENT T *release(void) {
    T *temp = ptr;
    ptr = nullptr;
    return temp;
  }

  DEVICE_INDEPENDENT void swap(Managed<T, use_new> &other) {
    swap(ptr, other.ptr);
  }

  DEVICE_INDEPENDENT const T &reference(void) const { return *ptr; }

  DEVICE_INDEPENDENT T &reference(void) { return *ptr; }

  DEVICE_INDEPENDENT operator T &(void) { return reference(); }

  DEVICE_INDEPENDENT operator const T &(void) const { return reference(); }
};

#endif
