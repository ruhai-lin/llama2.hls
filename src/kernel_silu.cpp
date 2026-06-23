#ifdef BUILD_DECODE_KERNEL

#ifndef LLAMA2_KERNEL_SILU_CPP_
#define LLAMA2_KERNEL_SILU_CPP_

#include "tensor.hpp"

#include <math.h>

namespace llama2 {

static inline float kernel_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

static inline float kernel_silu_scalar(float x) { return x * kernel_sigmoid(x); }

static void kernel_silu(Tensor1dFFNB& out, const Tensor1dFFNB& in) {
  for (int i = 0; i < kFFNDim; ++i) {
#pragma HLS PIPELINE II = 1
    out[i] = kernel_silu_scalar(in[i]);
  }
}

} // namespace llama2

#endif // LLAMA2_KERNEL_SILU_CPP_

#endif // BUILD_DECODE_KERNEL
