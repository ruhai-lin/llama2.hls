#ifdef BUILD_DECODE_KERNEL

#include "tensor.hpp"

#include <math.h>

namespace swan {

static void kernel_mul(Tensor1dFFNB& out, const Tensor1dFFNB& lhs,
                       const Tensor1dFFNB& rhs) {
  for (int i = 0; i < kFFNDim; ++i) {
#pragma HLS PIPELINE II = 1
    const float x = lhs[i];
    out[i] = (x / (1.0f + expf(-x))) * rhs[i];
  }
}

} // namespace swan

#endif // BUILD_DECODE_KERNEL
