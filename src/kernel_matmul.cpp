#ifdef BUILD_DECODE_KERNEL

#include "tensor.hpp"

#include <math.h>

namespace llama2 {

constexpr int kLinearMaxIn = kFFNDim;
constexpr int kLinearMaxOut = kFFNDim;
constexpr int kLinearMaxGroups = kFFNGroups;
typedef int8_t MatmulWeightWord __attribute__((vector_size(32)));

static void kernel_matmul(float* out, int out_size, const float* in,
                          int in_size, const int8_t* weight_q,
                          const float* weight_s) {
#pragma HLS INLINE off
  // -------------------------
  // Stage A: quantize the input vector.
  // -------------------------
  int8_t in_q[kLinearMaxIn];
  float in_s[kLinearMaxGroups];
  float weight_s_cache[kLinearMaxOut][kLinearMaxGroups];

#pragma HLS ARRAY_PARTITION variable = in_q cyclic factor = 32 dim = 1
#pragma HLS ARRAY_PARTITION variable = in_s complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weight_s_cache complete dim = 2

  const int group_count = in_size / kQuantGroupSize;

  for (int group = 0; group < in_size / kQuantGroupSize; ++group) {
    float max_abs = 0.0f;
    for (int i = 0; i < kQuantGroupSize; ++i) {
#pragma HLS PIPELINE II = 1
      const float val = fabsf(in[group * kQuantGroupSize + i]);
      if (val > max_abs) {
        max_abs = val;
      }
    }

    const float scale = max_abs / 127.0f;
    in_s[group] = scale;
    for (int i = 0; i < kQuantGroupSize; ++i) {
#pragma HLS PIPELINE II = 1
      const int col = group * kQuantGroupSize + i;
      float q = scale == 0.0f ? 0.0f : nearbyintf(in[col] / scale);
      if (q > 127.0f) {
        q = 127.0f;
      }
      if (q < -127.0f) {
        q = -127.0f;
      }
      in_q[col] = static_cast<int8_t>(q);
    }
  }

  // -------------------------
  // Stage B: cache weight scales on chip.
  // -------------------------
  for (int row = 0; row < out_size; ++row) {
    for (int group = 0; group < group_count; ++group) {
#pragma HLS PIPELINE II = 1
      weight_s_cache[row][group] = weight_s[row * group_count + group];
    }
  }

  const MatmulWeightWord* weight_words =
      reinterpret_cast<const MatmulWeightWord*>(weight_q);

  // -------------------------
  // Stage C: stream Q8 weights and keep dot products in int32.
  // -------------------------
  for (int row = 0; row < out_size; ++row) {
    int32_t dot_group[kLinearMaxGroups];
    float partial[kLinearMaxGroups];
    float sum12[kLinearMaxGroups / 2];
    float sum24[kLinearMaxGroups / 4];
    float sum48[kLinearMaxGroups / 8];

#pragma HLS ARRAY_PARTITION variable = dot_group complete dim = 1
#pragma HLS ARRAY_PARTITION variable = partial complete dim = 1
#pragma HLS ARRAY_PARTITION variable = sum12 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = sum24 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = sum48 complete dim = 1

    for (int group = 0; group < kLinearMaxGroups; ++group) {
#pragma HLS UNROLL
      dot_group[group] = 0;
    }

    for (int group = 0; group < group_count; ++group) {
#pragma HLS PIPELINE II = 1
      const MatmulWeightWord weight_vec = weight_words[row * group_count + group];
      int32_t dot = 0;

      for (int lane = 0; lane < kQuantGroupSize; ++lane) {
#pragma HLS UNROLL
        const int col = group * kQuantGroupSize + lane;
        dot += static_cast<int32_t>(in_q[col]) *
               static_cast<int32_t>(weight_vec[lane]);
      }

      dot_group[group] = dot;
    }

    // -------------------------
    // Stage D: dequantize and reduce the row result.
    // -------------------------
    for (int group = 0; group < kLinearMaxGroups; ++group) {
#pragma HLS UNROLL
      partial[group] =
          group < group_count
              ? static_cast<float>(dot_group[group]) * in_s[group] *
                    weight_s_cache[row][group]
              : 0.0f;
    }

    for (int i = 0; i < kLinearMaxGroups / 2; ++i) {
#pragma HLS UNROLL
      sum12[i] = partial[i * 2] + partial[i * 2 + 1];
    }
    for (int i = 0; i < kLinearMaxGroups / 4; ++i) {
#pragma HLS UNROLL
      sum24[i] = sum12[i * 2] + sum12[i * 2 + 1];
    }
    for (int i = 0; i < kLinearMaxGroups / 8; ++i) {
#pragma HLS UNROLL
      sum48[i] = sum24[i * 2] + sum24[i * 2 + 1];
    }

    out[row] = sum48[0] + sum48[1] + sum48[2];
  }
}

} // namespace llama2

#endif // BUILD_DECODE_KERNEL
