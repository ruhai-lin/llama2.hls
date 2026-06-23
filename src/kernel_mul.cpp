#ifdef BUILD_DECODE_KERNEL

#include "tensor.hpp"

#include <math.h>

namespace llama2 {

static void kernel_mul(Tensor1dFFNB& out, const Tensor1dFFNB& lhs,
                       const Tensor1dFFNB& rhs) {
  for (int i = 0; i < kFFNDim; ++i) {
#pragma HLS PIPELINE II = 1
    out[i] = lhs[i] * rhs[i];
  }
}

} // namespace llama2

#endif // BUILD_DECODE_KERNEL
