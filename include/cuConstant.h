#ifndef __cuConstant_cuh__
#define __cuConstant_cuh__

#include "cuBase.h"
#include "cuShared.h"

namespace internal {

template <typename T, const T &device_symbol> class ConstantImpl {
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

  __device__ inline cudaError_t fetch(T &host_mem) const {
    return cuAssert(cudaMemcpyFromSymbolAsync(&host_mem, symbol(), sizeof(T), 0,
                                              cudaMemcpyDeviceToDevice));
  }
};

} // namespace internal

template <typename T, const T &device_symbol>
class Constant : public internal::ConstantImpl<T, device_symbol> {};

template <typename T, const Segment<T> &device_symbol>
class Constant<Segment<T>, device_symbol>
    : public internal::ConstantImpl<Segment<T>, device_symbol> {
public:
  using Base = internal::ConstantImpl<Segment<T>, device_symbol>;

  inline cudaError_t commit(const T &host_mem) const {
    return this->Base::commit(reinterpret_cast<const Segment<T> &>(host_mem));
  }

  DEVICE_INDEPENDENT inline cudaError_t fetch(T &host_mem) const {
    return this->Base::fetch(reinterpret_cast<const Segment<T> &>(host_mem));
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

  inline cudaError_t commit() const { return this->commit(cache()); }

  DEVICE_INDEPENDENT inline cudaError_t fetch() const {
    return this->fetch(cache_mem);
  }
};

#endif
