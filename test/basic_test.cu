#include "cuManaged.h"
#include "cuShared.h"
#include <array>

#include <cooperative_groups.h>
#include <iostream>

template <typename T, std::size_t N>
__global__ void vec_add(std::array<T, N> &result, const std::array<T, N> &left,
                        const std::array<T, N> &right) {
  auto block = cooperative_groups::this_thread_block();
  if (block.thread_rank() < N) {
    result[block.thread_rank()] =
        left[block.thread_rank()] + right[block.thread_rank()];
  }
}

static constexpr std::array<double, 3> initial = {2, 4, 3};
static constexpr std::array<double, 3> displacement = {3, 6, 4};

int main(int argc, char *argv[]) {
  Managed<std::array<double, 3>> left(initial);
  Managed<std::array<double, 3>> right(displacement);
  Managed<std::array<double, 3>, false> result;
  vec_add<<<1, 3>>>(result.reference(), left.reference(), right.reference());
  cudaDeviceSynchronize();
  assert(result.reference()[0] == 5);
  assert(result.reference()[1] == 10);
  assert(result.reference()[2] == 7);
  std::cout << "PASSED" << std::endl;
  return 0;
}
