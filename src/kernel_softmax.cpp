#ifdef BUILD_DECODE_KERNEL

#include "tensor.hpp"

#include <math.h>

namespace swan {

static void kernel_softmax(Tensor1dQKSM& out, const Tensor1dQKSM& in,
                           int size) {
  float max_val = in[0];
  for (int i = 1; i < size; ++i) {
#pragma HLS PIPELINE II = 1
    if (in[i] > max_val) {
      max_val = in[i];
    }
  }

  float sum = 0.0f;
  for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II = 1
    const float v = expf(in[i] - max_val);
    out[i] = v;
    sum += v;
  }

  for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II = 1
    out[i] /= sum;
  }
}

} // namespace swan

#endif // BUILD_DECODE_KERNEL
