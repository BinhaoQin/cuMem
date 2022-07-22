#ifndef __cuConstant_cuh__
#define __cuConstant_cuh__

#include "cuBase.h"
#include "cuShared.h"

template <typename T, const T &device_symbol> class Constant {
public:
  DEVICE_INDEPENDENT static constexpr const T &symbol(void) {
    return device_symbol;
  }

  DEVICE_INDEPENDENT static const T &reference(void) {
    const T *address = nullptr;
    cuAssert(cudaGetSymbolAddress((void **)(&address), symbol()));
    return *address;
  }

  static inline cudaError_t commit(const T &host_mem) {
    return cuAssert(cudaMemcpyToSymbolAsync(symbol(), &host_mem, sizeof(T)));
  }

  static inline cudaError_t fetch(T &host_mem) {
    return cuAssert(cudaMemcpyFromSymbolAsync(&host_mem, symbol(), sizeof(T)));
  }

  __device__ static inline cudaError_t fetch(T &host_mem) {
    return cuAssert(cudaMemcpyFromSymbolAsync(&host_mem, symbol(), sizeof(T), 0,
                                              cudaMemcpyDeviceToDevice));
  }
};

template <typename U, const Segment<U> &device_symbol>
class Constant<Segment<U>, device_symbol> {
protected:
  using T = Segment<U>;

public:
  DEVICE_INDEPENDENT static constexpr const T &symbol(void) {
    return device_symbol;
  }

  DEVICE_INDEPENDENT static const T &reference(void) {
    const T *address = nullptr;
    cuAssert(cudaGetSymbolAddress((void **)(&address), symbol()));
    return *address;
  }

  static inline cudaError_t commit(const T &host_mem) {
    return cuAssert(cudaMemcpyToSymbolAsync(symbol(), &host_mem, sizeof(T)));
  }

  static inline cudaError_t fetch(T &host_mem) {
    return cuAssert(cudaMemcpyFromSymbolAsync(&host_mem, symbol(), sizeof(T)));
  }

  __device__ static inline cudaError_t fetch(T &host_mem) {
    return cuAssert(cudaMemcpyFromSymbolAsync(&host_mem, symbol(), sizeof(T), 0,
                                              cudaMemcpyDeviceToDevice));
  }

  static inline cudaError_t commit(const U &host_mem) {
    return commit(reinterpret_cast<const Segment<T> &>(host_mem));
  }

  DEVICE_INDEPENDENT static inline cudaError_t fetch(U &host_mem) {
    return fetch(reinterpret_cast<const Segment<T> &>(host_mem));
  }
};

template <typename T, const T &device_symbol>
class ConstantCached : public Constant<T, device_symbol> {
protected:
  T cache_mem;

public:
  using Base = Constant<T, device_symbol>;

  template <typename... Args>
  DEVICE_INDEPENDENT ConstantCached(Args &&...args) : cache_mem(args...) {}

  DEVICE_INDEPENDENT const T &cache(void) const { return cache_mem; }

  DEVICE_INDEPENDENT T &cache(void) { return cache_mem; }

  inline cudaError_t commit() const { return Base::commit(cache()); }

  DEVICE_INDEPENDENT inline cudaError_t fetch() const {
    return Base::fetch(cache_mem);
  }
};

#endif
