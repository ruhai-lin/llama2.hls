#ifdef BUILD_DECODE_KERNEL

#include "tensor.hpp"

namespace swan {

static void kernel_rope(Tensor1d& q_out, Tensor1d& k_out,
                        const Tensor1d& q_in, const Tensor1d& k_in,
                        const float* cos_table, const float* sin_table,
                        int pos) {
  constexpr int kHeadDim = kDim / kNumHeads;
  for (int head = 0; head < kNumHeads; ++head) {
    const int head_begin = head * kHeadDim;
    for (int i = 0; i < kHeadDim / 2; ++i) {
#pragma HLS PIPELINE II = 1
      const int i0 = head_begin + i * 2;
      const int i1 = i0 + 1;
      const float c = cos_table[pos * kSinCosTable + i];
      const float s = sin_table[pos * kSinCosTable + i];
      const float q0 = q_in[i0];
      const float q1 = q_in[i1];
      const float k0 = k_in[i0];
      const float k1 = k_in[i1];

      q_out[i0] = q0 * c - q1 * s;
      q_out[i1] = q0 * s + q1 * c;
      k_out[i0] = k0 * c - k1 * s;
      k_out[i1] = k0 * s + k1 * c;
    }
  }
}

} // namespace swan

#endif // BUILD_DECODE_KERNEL
