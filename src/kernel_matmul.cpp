#ifdef BUILD_DECODE_KERNEL

#include "tensor.hpp"

namespace swan {

static void kernel_matmul(float* out, int out_size, const float* in,
                          int in_size, const float* weight) {
  for (int row = 0; row < out_size; ++row) {
    float sum = 0.0f;
    for (int col = 0; col < in_size; ++col) {
#pragma HLS PIPELINE II = 1
      sum += weight[row * in_size + col] * in[col];
    }
    out[row] = sum;
  }
}

} // namespace swan

#endif // BUILD_DECODE_KERNEL
