#ifdef BUILD_DECODE_KERNEL

#include "tensor.hpp"

#include <math.h>

namespace swan {

static void kernel_rmsnorm(Tensor1d& out, const Tensor1d& in,
                           const float* weight) {
  float sum = 0.0f;
  for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
    sum += in[i] * in[i];
  }

  const float norm = 1.0f / sqrtf(sum / kDim + 1e-5f);
  for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
    out[i] = in[i] * norm * weight[i];
  }
}

} // namespace swan

#endif // BUILD_DECODE_KERNEL
