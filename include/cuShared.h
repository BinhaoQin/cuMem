#ifndef __cuShared_cuh__
#define __cuShared_cuh__

#include "cuBase.h"
#include <new>

// Segment<T> is a template class that emulates the memory layout of T
// but strips off T's type info and manually add it later on.
// The purpose is solely to bybass constructor.
template <typename T> class alignas(alignof(T)) Segment {
protected:
  char __data[sizeof(T)];

public:
  template <typename... Args>
  DEVICE_INDEPENDENT void init(const Args &...args) {
    new (__data) T(args...);
  }

  DEVICE_INDEPENDENT T *operator->(void) {
    return reinterpret_cast<T *>(__data);
  }

  DEVICE_INDEPENDENT const T *operator->(void) const {
    return reinterpret_cast<const T *>(__data);
  }

  DEVICE_INDEPENDENT inline T &reference(void) {
    return *reinterpret_cast<T *>(__data);
  }

  DEVICE_INDEPENDENT inline const T &reference(void) const {
    return *reinterpret_cast<const T *>(__data);
  }

  DEVICE_INDEPENDENT operator T &(void) { return reference(); }

  DEVICE_INDEPENDENT operator const T &(void) const { return reference(); }
};

template <typename T> class DeferredInit {
public:
  Segment<T> segment;
};

#endif
