#ifndef __cuCopy_h__
#define __cuCopy_h__

#include "cuBase.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

template <class TyGroup, typename TyElem, typename TySizeT>
inline __device__ void group_copy_sync(const TyGroup &group, TyElem &_dst,
                                       const TyElem &_src) {
  assert(&_dst > &_src || &_dst + sizeof(TyElem) < &_src);
  cg::memcpy_async(group, &_dst, &_src, sizeof(TyElem));
  cg::wait(group);
}

#endif
