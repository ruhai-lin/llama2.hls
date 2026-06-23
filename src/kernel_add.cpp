#ifdef BUILD_DECODE_KERNEL

#include "tensor.hpp"

namespace llama2 {

static void kernel_add(Tensor1d& out, const Tensor1d& lhs,
                       const Tensor1d& rhs) {
  for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
    out[i] = lhs[i] + rhs[i];
  }
}

} // namespace llama2

#endif // BUILD_DECODE_KERNEL
