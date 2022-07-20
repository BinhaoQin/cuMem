#ifndef __cuEvent_h__
#define __cuEvent_h__

#include "cuBase.h"
#include <utility>

// move semantics

class Event {
protected:
  cudaEvent_t _event;

public:
  DEVICE_INDEPENDENT Event(unsigned int flags = cudaEventDefault |
                                                cudaEventBlockingSync |
                                                cudaEventDisableTiming)
      : _event(nullptr) {
    cuAssert(cudaEventCreateWithFlags(&_event, flags));
  }

  inline cudaEvent_t get(void) const { return _event; }

  DEVICE_INDEPENDENT cudaError_t reset(cudaEvent_t event = nullptr) {
    if (_event)
      return cuAssert(cudaEventDestroy(_event));
    _event = event;
    return cudaSuccess;
  }

  DEVICE_INDEPENDENT inline cudaEvent_t release(void) {
    cudaEvent_t result = _event;
    this->reset();
    return result;
  }

  DEVICE_INDEPENDENT Event(Event &&event) : _event(event._event) {
    event.reset();
  }

  DEVICE_INDEPENDENT inline cudaError_t replace(Event &&event) {
    cudaError_t result = cuAssert(this->reset(event._event));
    event.reset();
    return result;
  }

  DEVICE_INDEPENDENT inline cudaError_t
  reuse(unsigned int flags = cudaEventDefault | cudaEventBlockingSync |
                             cudaEventDisableTiming) {
    this->reset();
    return cuAssert(cudaEventCreateWithFlags(&_event, flags));
  }

  DEVICE_INDEPENDENT inline cudaError_t record(cudaStream_t stream = 0) const {
    return cuAssert(cudaEventRecord(_event, stream));
  }

  inline cudaError_t wait(void) const {
    return cuAssert(cudaEventSynchronize(_event));
  }

  inline cudaError_t query(void) const { return cudaEventQuery(_event); }

  inline float elapsedMillisFrom(const Event &event) const {
    float ms = 0;
    cuAssert(cudaEventElapsedTime(&ms, event._event, _event));
    return ms;
  }
};

#endif
