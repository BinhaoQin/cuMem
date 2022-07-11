#ifndef __cuConstant_cuh__
#define __cuConstant_cuh__

#include "cuBase.h"
#include "cuShared.h"

template <typename T, const T &device_symbol> class Constant {
public:
  DEVICE_INDEPENDENT constexpr const T &symbol(void) const {
    return device_symbol;
  }

  DEVICE_INDEPENDENT const T &reference(void) const {
    const T *address = nullptr;
    cuAssert(cudaGetSymbolAddress((void **)(&address), symbol()));
    return *address;
  }

  inline cudaError_t commit(const T &host_mem) const {
    return cuAssert(cudaMemcpyToSymbolAsync(symbol(), &host_mem, sizeof(T)));
  }

  inline cudaError_t fetch(T &host_mem) const {
    return cuAssert(cudaMemcpyFromSymbolAsync(&host_mem, symbol(), sizeof(T)));
  }

  __device__ inline cudaError_t commit(const T &host_mem) const {
    return cuAssert(cudaMemcpyToSymbolAsync(symbol(), &host_mem, sizeof(T), 0,
                                            cudaMemcpyDeviceToDevice));
  }

  __device__ inline cudaError_t fetch(T &host_mem) const {
    return cuAssert(cudaMemcpyFromSymbolAsync(&host_mem, symbol(), sizeof(T), 0,
                                              cudaMemcpyDeviceToDevice));
  }
};

template <typename T, const T &device_symbol>
class ConstantCached : public Constant<T, device_symbol> {
protected:
  T cache_mem;

public:
  template <typename... Args>
  DEVICE_INDEPENDENT ConstantCached(Args &&...args) : cache_mem(args...) {}

  DEVICE_INDEPENDENT const T &cache(void) const { return cache_mem; }

  DEVICE_INDEPENDENT T &cache(void) { return cache_mem; }

  inline cudaError_t commit() const {
    return cuAssert(
        cudaMemcpyToSymbolAsync(this->symbol(), &cache(), sizeof(T)));
  }

  inline cudaError_t fetch() const {
    return cuAssert(
        cudaMemcpyFromSymbolAsync(&cache(), this->symbol(), sizeof(T)));
  }

  __device__ inline cudaError_t commit() const {
    return cuAssert(cudaMemcpyToSymbolAsync(this->symbol(), &cache(), sizeof(T),
                                            0, cudaMemcpyDeviceToDevice));
  }

  __device__ inline cudaError_t fetch() const {
    return cuAssert(cudaMemcpyFromSymbolAsync(
        &cache(), this->symbol(), sizeof(T), 0, cudaMemcpyDeviceToDevice));
  }
};

#endif
