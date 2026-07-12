#ifdef BUILD_DECODE_KERNEL

#include "tensor.hpp"

#include <float.h>
#include <math.h>
#include <stdint.h>

namespace llama2 {
namespace {

constexpr int kHeadDim = kDim / kNumHeads;
constexpr int kAccStages = 8;

using Q8ActivationBuffer = int8_t[kFFNDim];
using Q8ActivationScaleBuffer = float[kFFNGroups];
using GemvBlockScores = float[kPackedRows];

inline int rms_idx(int layer, int i) { return layer * kDim + i; }
inline int cache_idx(int layer, int pos, int i) {
  return (layer * kSeqLen + pos) * kDim + i;
}

static inline float fadd(float a, float b) {
  float y = a + b;
#pragma HLS BIND_OP variable = y op = fadd impl = fulldsp latency = 6
  return y;
}

static inline void clear_acc(float acc[kAccStages]) {
  for (int stage = 0; stage < kAccStages; ++stage) {
#pragma HLS UNROLL
    acc[stage] = 0.0f;
  }
}

static inline float reduce_acc(const float acc[kAccStages]) {
  const float sum01 = fadd(acc[0], acc[1]);
  const float sum23 = fadd(acc[2], acc[3]);
  const float sum45 = fadd(acc[4], acc[5]);
  const float sum67 = fadd(acc[6], acc[7]);
  return fadd(fadd(sum01, sum23), fadd(sum45, sum67));
}

template <int Size>
static void quantize(Q8ActivationBuffer& out,
                     Q8ActivationScaleBuffer& scales,
                     const float (&in)[Size]) {
  for (int group = 0; group < Size / kQuantGroupSize; ++group) {
    float max_abs = 0.0f;
    for (int lane = 0; lane < kQuantGroupSize; ++lane) {
#pragma HLS PIPELINE II = 1
      const float value = fabsf(in[group * kQuantGroupSize + lane]);
      if (value > max_abs) {
        max_abs = value;
      }
    }

    const float scale = max_abs / 127.0f;
    scales[group] = scale;
    for (int lane = 0; lane < kQuantGroupSize; ++lane) {
#pragma HLS PIPELINE II = 1
      const int index = group * kQuantGroupSize + lane;
      float value = scale == 0.0f ? 0.0f : nearbyintf(in[index] / scale);
      if (value > 127.0f) {
        value = 127.0f;
      } else if (value < -127.0f) {
        value = -127.0f;
      }
      out[index] = static_cast<int8_t>(value);
    }
  }
}

static inline float unpack_scale(const PackedParameterWord& word, int row) {
  union {
    uint32_t bits;
    float value;
  } converter;
  const int base = row * sizeof(float);
  converter.bits = static_cast<uint32_t>(word[base]) |
                   (static_cast<uint32_t>(word[base + 1]) << 8) |
                   (static_cast<uint32_t>(word[base + 2]) << 16) |
                   (static_cast<uint32_t>(word[base + 3]) << 24);
  return converter.value;
}

template <int WeightBase>
static inline int32_t dot32(const PackedParameterWord& weights,
                            const int8_t input[kQuantGroupSize]) {
  int32_t level0[kQuantGroupSize];
  int32_t level1[kQuantGroupSize / 2];
  int32_t level2[kQuantGroupSize / 4];
  int32_t level3[kQuantGroupSize / 8];
  int32_t level4[kQuantGroupSize / 16];
#pragma HLS ARRAY_PARTITION variable = level0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = level1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = level2 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = level3 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = level4 complete dim = 1

  for (int lane = 0; lane < kQuantGroupSize; ++lane) {
#pragma HLS UNROLL
    level0[lane] =
        static_cast<int32_t>(
            static_cast<int8_t>(weights[WeightBase + lane])) *
        static_cast<int32_t>(input[lane]);
  }
  for (int lane = 0; lane < kQuantGroupSize / 2; ++lane) {
#pragma HLS UNROLL
    level1[lane] = level0[lane * 2] + level0[lane * 2 + 1];
  }
  for (int lane = 0; lane < kQuantGroupSize / 4; ++lane) {
#pragma HLS UNROLL
    level2[lane] = level1[lane * 2] + level1[lane * 2 + 1];
  }
  for (int lane = 0; lane < kQuantGroupSize / 8; ++lane) {
#pragma HLS UNROLL
    level3[lane] = level2[lane * 2] + level2[lane * 2 + 1];
  }
  for (int lane = 0; lane < kQuantGroupSize / 16; ++lane) {
#pragma HLS UNROLL
    level4[lane] = level3[lane * 2] + level3[lane * 2 + 1];
  }
  return level4[0] + level4[1];
}

static void q8_gemv_engine(const PackedParameterWord* packed_params,
                           int matrix_word_offset, int group_count,
                           const Q8ActivationBuffer& activation,
                           const Q8ActivationScaleBuffer& activation_scales,
                           GemvBlockScores& scores) {
#pragma HLS INLINE off
  PackedParameterWord block_words[kFFNGroups * (1 + kPackedRows / 2)];
  float acc[kPackedRows][kAccStages];
#pragma HLS ARRAY_PARTITION variable = activation cyclic factor = 32 dim = 1
#pragma HLS ARRAY_PARTITION variable = activation_scales complete dim = 1
#pragma HLS ARRAY_PARTITION variable = scores complete dim = 1
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0
#pragma HLS BIND_STORAGE variable = block_words type = ram_2p impl = uram

  const int block_word_count = group_count * (1 + kPackedRows / 2);
  for (int word = 0; word < block_word_count; ++word) {
#pragma HLS PIPELINE II = 1
    block_words[word] = packed_params[matrix_word_offset + word];
  }

  for (int row = 0; row < kPackedRows; ++row) {
#pragma HLS UNROLL
    clear_acc(acc[row]);
  }

  for (int group = 0; group < group_count; ++group) {
    const PackedParameterWord scale_word =
        block_words[group * (1 + kPackedRows / 2)];
    float weight_scales[kPackedRows];
    int8_t input_group[kQuantGroupSize];
#pragma HLS ARRAY_PARTITION variable = weight_scales complete dim = 1
#pragma HLS ARRAY_PARTITION variable = input_group complete dim = 1
    for (int row = 0; row < kPackedRows; ++row) {
#pragma HLS UNROLL
      weight_scales[row] = unpack_scale(scale_word, row);
    }
    for (int lane = 0; lane < kQuantGroupSize; ++lane) {
#pragma HLS UNROLL
      input_group[lane] = activation[group * kQuantGroupSize + lane];
    }

    // Each iteration owns a distinct row pair, so acc has no pair-to-pair RAW.
    for (int pair = 0; pair < kPackedRows / 2; ++pair) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter false
      const PackedParameterWord weight_word =
          block_words[group * (1 + kPackedRows / 2) + 1 + pair];
      const int row0 = pair * 2;
      const int row1 = row0 + 1;
      const int stage = group & (kAccStages - 1);
      const int32_t dot0 = dot32<0>(weight_word, input_group);
      const int32_t dot1 =
          dot32<kQuantGroupSize>(weight_word, input_group);
      const float input_scale = activation_scales[group];
      const float combined_scale0 = input_scale * weight_scales[row0];
      const float combined_scale1 = input_scale * weight_scales[row1];
      const float partial0 = static_cast<float>(dot0) * combined_scale0;
      const float partial1 = static_cast<float>(dot1) * combined_scale1;
      acc[row0][stage] = fadd(acc[row0][stage], partial0);
      acc[row1][stage] = fadd(acc[row1][stage], partial1);
    }
  }

  for (int row = 0; row < kPackedRows; ++row) {
#pragma HLS PIPELINE II = 1
    scores[row] = reduce_acc(acc[row]);
  }
}

static void q8_linear(float* output, int output_rows,
                      const Q8ActivationBuffer& activation,
                      const Q8ActivationScaleBuffer& activation_scales,
                      const PackedParameterWord* packed_params,
                      int matrix_word_offset, int group_count) {
#pragma HLS INLINE
  // Constant 9/24 propagation makes Vitis HLS clone the physical engine.
  volatile int runtime_group_count = group_count;
  const int block_word_count = group_count * (1 + kPackedRows / 2);
  for (int block = 0; block < output_rows / kPackedRows; ++block) {
    GemvBlockScores scores;
#pragma HLS ARRAY_PARTITION variable = scores complete dim = 1
    q8_gemv_engine(packed_params,
                   matrix_word_offset + block * block_word_count,
                   runtime_group_count, activation, activation_scales, scores);
    for (int row = 0; row < kPackedRows; ++row) {
#pragma HLS UNROLL
      output[block * kPackedRows + row] = scores[row];
    }
  }
}

static void load_token(Tensor1d& hidden, int token,
                       const float* tok_emb_table) {
  for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
    hidden[i] = tok_emb_table[token * kDim + i];
  }
}

static void rmsnorm(Tensor1d& out, const Tensor1d& in, const float* weight) {
  float acc[kAccStages];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 1
  clear_acc(acc);

  // A stage is revisited after 8 cycles, beyond the 6-cycle FP32 adder latency.
  for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter false
    const int stage = i & (kAccStages - 1);
    acc[stage] = fadd(acc[stage], in[i] * in[i]);
  }

  const float sum = reduce_acc(acc);
  const float norm = 1.0f / sqrtf(sum / kDim + 1e-5f);
  for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
    out[i] = in[i] * norm * weight[i];
  }
}

static void add(Tensor1d& out, const Tensor1d& lhs, const Tensor1d& rhs) {
  for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
    out[i] = lhs[i] + rhs[i];
  }
}

static void add_inplace(Tensor1d& out, const Tensor1d& rhs) {
  for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
    out[i] += rhs[i];
  }
}

static void rope(Tensor1d& q_out, Tensor1d& k_out, const Tensor1d& q_in,
                 const Tensor1d& k_in, const float* cos_table,
                 const float* sin_table, int pos) {
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

static void softmax(Tensor1dQKSM& out, const Tensor1dQKSM& in, int size) {
  float max_val = in[0];
  for (int i = 1; i < size; ++i) {
#pragma HLS PIPELINE II = 1
    if (in[i] > max_val) {
      max_val = in[i];
    }
  }

  float acc[kAccStages];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 1
  clear_acc(acc);

  // A stage is revisited after 8 cycles, beyond the 6-cycle FP32 adder latency.
  for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter false
    const float v = expf(in[i] - max_val);
    const int stage = i & (kAccStages - 1);
    out[i] = v;
    acc[stage] = fadd(acc[stage], v);
  }

  const float sum = reduce_acc(acc);
  for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II = 1
    out[i] /= sum;
  }
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

static void silu_mul(Tensor1dFFNB& out, const Tensor1dFFNB& gate,
                     const Tensor1dFFNB& up) {
  for (int i = 0; i < kFFNDim; ++i) {
#pragma HLS PIPELINE II = 1
    out[i] = silu(gate[i]) * up[i];
  }
}

static void attn(Tensor1d& hidden, int layer, int pos,
                 const float* rms_att_w,
                 const PackedParameterWord* packed_params,
                 const float* cos_table, const float* sin_table,
                 float* k_cache, float* v_cache) {
#pragma HLS INLINE

  Tensor1d attn_input;
  Tensor1d attn_norm;
  Tensor1d q;
  Tensor1d k;
  Tensor1d v;
  Tensor1d q_rot;
  Tensor1d k_rot;
  Tensor1dQKSM qk;
  Tensor1dQKSM sm;
  Tensor1d attn_val;
  Tensor1d attn_out;
  Q8ActivationBuffer attn_norm_q8;
  Q8ActivationScaleBuffer attn_norm_scales;
  Q8ActivationBuffer attn_value_q8;
  Q8ActivationScaleBuffer attn_value_scales;
#pragma HLS ARRAY_PARTITION variable = attn_norm_q8 cyclic factor = 32 dim = 1
#pragma HLS ARRAY_PARTITION variable = attn_norm_scales complete dim = 1
#pragma HLS ARRAY_PARTITION variable = attn_value_q8 cyclic factor = 32 dim = 1
#pragma HLS ARRAY_PARTITION variable = attn_value_scales complete dim = 1

  for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
    attn_input[i] = hidden[i];
  }

  rmsnorm(attn_norm, attn_input, &rms_att_w[rms_idx(layer, 0)]);
  quantize(attn_norm_q8, attn_norm_scales, attn_norm);

  q8_linear(q, kDim, attn_norm_q8, attn_norm_scales, packed_params,
            kPackedAttnWqWordOffset + layer * kPackedAttnLayerWords,
            kDimGroups);

  q8_linear(k, kDim, attn_norm_q8, attn_norm_scales, packed_params,
            kPackedAttnWkWordOffset + layer * kPackedAttnLayerWords,
            kDimGroups);

  q8_linear(v, kDim, attn_norm_q8, attn_norm_scales, packed_params,
            kPackedAttnWvWordOffset + layer * kPackedAttnLayerWords,
            kDimGroups);

  rope(q_rot, k_rot, q, k, cos_table, sin_table, pos);

  for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
    k_cache[cache_idx(layer, pos, i)] = k_rot[i];
    v_cache[cache_idx(layer, pos, i)] = v[i];
    attn_val[i] = 0.0f;
  }

  const float scale = 1.0f / sqrtf(static_cast<float>(kHeadDim));
  for (int head = 0; head < kNumHeads; ++head) {
    const int head_begin = head * kHeadDim;
    const int head_end = head_begin + kHeadDim;

    for (int t = 0; t <= pos; ++t) {
      float acc[kAccStages];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 1
      clear_acc(acc);

      // The 8-way accumulator spacing covers the 6-cycle FP32 adder latency.
      for (int i = head_begin; i < head_end; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter false
        const int stage = i & (kAccStages - 1);
        acc[stage] = fadd(acc[stage], q_rot[i] * k_cache[cache_idx(layer, t, i)]);
      }
      const float sum = reduce_acc(acc);
      qk[t] = sum * scale;
    }
    softmax(sm, qk, pos + 1);

    for (int i = head_begin; i < head_end; ++i) {
      float acc[kAccStages];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 1
      clear_acc(acc);

      // The 8-way accumulator spacing covers the 6-cycle FP32 adder latency.
      for (int t = 0; t <= pos; ++t) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter false
        const int stage = t & (kAccStages - 1);
        acc[stage] = fadd(acc[stage], sm[t] * v_cache[cache_idx(layer, t, i)]);
      }
      const float sum = reduce_acc(acc);
      attn_val[i] = sum;
    }
  }

  quantize(attn_value_q8, attn_value_scales, attn_val);
  q8_linear(attn_out, kDim, attn_value_q8, attn_value_scales, packed_params,
            kPackedAttnWoWordOffset + layer * kPackedAttnLayerWords,
            kDimGroups);

  add(hidden, attn_input, attn_out);
}

static void ffn(Tensor1d& hidden, int layer, const float* rms_ffn_w,
                const PackedParameterWord* packed_params) {
#pragma HLS INLINE

  Tensor1d ffn_norm;
  Tensor1dFFNB w1;
  Tensor1dFFNB w3;
  Tensor1dFFNB dot;
  Tensor1d out;
  Q8ActivationBuffer ffn_norm_q8;
  Q8ActivationScaleBuffer ffn_norm_scales;
  Q8ActivationBuffer ffn_product_q8;
  Q8ActivationScaleBuffer ffn_product_scales;
#pragma HLS ARRAY_PARTITION variable = ffn_norm_q8 cyclic factor = 32 dim = 1
#pragma HLS ARRAY_PARTITION variable = ffn_norm_scales complete dim = 1
#pragma HLS ARRAY_PARTITION variable = ffn_product_q8 cyclic factor = 32 dim = 1
#pragma HLS ARRAY_PARTITION variable = ffn_product_scales complete dim = 1

  rmsnorm(ffn_norm, hidden, &rms_ffn_w[rms_idx(layer, 0)]);
  quantize(ffn_norm_q8, ffn_norm_scales, ffn_norm);

  q8_linear(w1, kFFNDim, ffn_norm_q8, ffn_norm_scales, packed_params,
            kPackedFFNW1WordOffset + layer * kPackedFFNALayerWords,
            kDimGroups);

  q8_linear(w3, kFFNDim, ffn_norm_q8, ffn_norm_scales, packed_params,
            kPackedFFNW3WordOffset + layer * kPackedFFNALayerWords,
            kDimGroups);

  silu_mul(dot, w1, w3);
  quantize(ffn_product_q8, ffn_product_scales, dot);

  q8_linear(out, kDim, ffn_product_q8, ffn_product_scales, packed_params,
            kPackedFFNW2WordOffset + layer * kPackedFFNBLayerWords,
            kFFNGroups);

  add_inplace(hidden, out);
}

static uint32_t lm_head(const Tensor1d& hidden,
                        const PackedParameterWord* packed_params,
                        const float* rms_final) {
#pragma HLS INLINE

  Tensor1d final_norm;
  Q8ActivationBuffer final_norm_q8;
  Q8ActivationScaleBuffer final_norm_scales;
#pragma HLS ARRAY_PARTITION variable = final_norm_q8 cyclic factor = 32 dim = 1
#pragma HLS ARRAY_PARTITION variable = final_norm_scales complete dim = 1

  rmsnorm(final_norm, hidden, rms_final);
  quantize(final_norm_q8, final_norm_scales, final_norm);

  float best_score = -FLT_MAX;
  uint32_t best_token = 0;
  for (int block = 0; block < kVocabSize / kPackedRows; ++block) {
    GemvBlockScores scores;
#pragma HLS ARRAY_PARTITION variable = scores complete dim = 1
    q8_linear(scores, kPackedRows, final_norm_q8, final_norm_scales,
              packed_params,
              kPackedTokWordOffset +
                  block * kDimGroups * (1 + kPackedRows / 2),
              kDimGroups);
    for (int row = 0; row < kPackedRows; ++row) {
      if (scores[row] > best_score) {
        best_score = scores[row];
        best_token = static_cast<uint32_t>(block * kPackedRows + row);
      }
    }
  }
  return best_token;
}

} // namespace
} // namespace llama2

using namespace llama2;

extern "C" {

void decode(int token, int pos, const float* tok_emb_table,
            const PackedParameterWord* packed_params,
            const float* rms_att_w, const float* rms_ffn_w,
            const float* rms_final, const float* cos_table,
            const float* sin_table, float* k_cache, float* v_cache,
            uint32_t* next_token) {
// -------------------------
// Stage 0: AXI interfaces.
// -------------------------
#pragma HLS INTERFACE m_axi port = tok_emb_table bundle = gmem
#pragma HLS INTERFACE m_axi port = packed_params bundle = gmem \
    max_read_burst_length = 256 num_read_outstanding = 2
#pragma HLS INTERFACE m_axi port = rms_att_w bundle = gmem
#pragma HLS INTERFACE m_axi port = rms_ffn_w bundle = gmem
#pragma HLS INTERFACE m_axi port = rms_final bundle = gmem
#pragma HLS INTERFACE m_axi port = cos_table bundle = gmem
#pragma HLS INTERFACE m_axi port = sin_table bundle = gmem
#pragma HLS INTERFACE m_axi port = k_cache bundle = gmem
#pragma HLS INTERFACE m_axi port = v_cache bundle = gmem
#pragma HLS INTERFACE m_axi port = next_token bundle = gmem
#pragma HLS ALLOCATION function instances = q8_gemv_engine limit = 1

  Tensor1d hidden;
  load_token(hidden, token, tok_emb_table);

  for (int layer = 0; layer < kNumLayers; ++layer) {
    attn(hidden, layer, pos, rms_att_w, packed_params, cos_table, sin_table,
         k_cache, v_cache);
    ffn(hidden, layer, rms_ffn_w, packed_params);
  }

  *next_token = lm_head(hidden, packed_params, rms_final);
}

} // extern "C"

#else

#include "decode.hpp"

#include <cmath>
#include <cstring>

namespace llama2 {

void Decode(int tok, int pos, const Tensor1d& ctx_input,
            Tensor3dCache& ctx_k_cache, Tensor3dCache& ctx_v_cache,
            Tensor1d& ctx_final_norm, Tensor1dLogits& ctx_logits,
            int& next_token,
            const Weights& w
#ifndef USE_CPU_ONLY
            ,
            cl::CommandQueue q, cl::Kernel decode_kernel, uint32_t* ptr_next,
            cl::Buffer buffer_next
#endif // USE_CPU_ONLY
) {
#ifndef USE_CPU_ONLY
  (void)ctx_input;
  (void)ctx_k_cache;
  (void)ctx_v_cache;
  (void)ctx_final_norm;
  (void)ctx_logits;
  (void)w;

  decode_kernel.setArg(0, tok);
  decode_kernel.setArg(1, pos);
  q.enqueueTask(decode_kernel);
  q.enqueueReadBuffer(buffer_next, CL_FALSE, 0, sizeof(uint32_t), ptr_next);
  q.finish();
  next_token = static_cast<int>(*ptr_next);
#else
  (void)tok;
  (void)next_token;
  static Context ctx;

  const int head_dim = kDim / kNumHeads;
  const float norm = 1 / std::sqrt(head_dim);

  Tensor1d attn_input;
  for (int i_layer = 0; i_layer < kNumLayers; ++i_layer) {
    if (i_layer == 0) {
      CopyTensor1d(attn_input, ctx_input);
    } else {
      CopyTensor1d(attn_input, ctx.ffn_res[i_layer - 1]);
    }

    RMSNorm(ctx.attn_norm[i_layer], attn_input, w.rms_att_w[i_layer]);
    Matmul(ctx.attn_wqx[i_layer], ctx.attn_norm[i_layer], w.attn_wq[i_layer],
           w.attn_wq_s[i_layer]);
    Matmul(ctx.attn_wkx[i_layer], ctx.attn_norm[i_layer], w.attn_wk[i_layer],
           w.attn_wk_s[i_layer]);
    Matmul(ctx.attn_wvx[i_layer], ctx.attn_norm[i_layer], w.attn_wv[i_layer],
           w.attn_wv_s[i_layer]);

    for (int head = 0; head < kNumHeads; ++head) {
      RoPE(ctx.attn_q_r[i_layer], ctx.attn_k_r[i_layer], ctx.attn_wqx[i_layer],
           ctx.attn_wkx[i_layer], w.cos_table[pos], w.sin_table[pos],
           head * head_dim, head_dim);
    }

    CopyTensor1d(ctx_k_cache[i_layer][pos], ctx.attn_k_r[i_layer]);
    CopyTensor1d(ctx_v_cache[i_layer][pos], ctx.attn_wvx[i_layer]);

    for (int i_head = 0; i_head < kNumHeads; ++i_head) {
      int head_begin = i_head * head_dim;
      int head_end = (i_head + 1) * head_dim;
      MutmulRanged(ctx.attn_qk[i_layer], ctx.attn_q_r[i_layer],
                   ctx_k_cache[i_layer], 0, pos + 1, head_begin, head_end);
      Mul(ctx.attn_qk[i_layer], ctx.attn_qk[i_layer], norm);
      Softmax(ctx.attn_sm[i_layer], ctx.attn_qk[i_layer], pos + 1);
      MutmulRangedTranspose(ctx.attn_val[i_layer], ctx.attn_sm[i_layer],
                            ctx_v_cache[i_layer], head_begin, head_end, 0,
                            pos + 1);
    }

    Matmul(ctx.attn_out[i_layer], ctx.attn_val[i_layer], w.attn_wo[i_layer],
           w.attn_wo_s[i_layer]);
    Add(ctx.attn_res[i_layer], attn_input, ctx.attn_out[i_layer]);
    RMSNorm(ctx.ffn_norm[i_layer], ctx.attn_res[i_layer], w.rms_ffn_w[i_layer]);
    Matmul(ctx.ffn_w1x[i_layer], ctx.ffn_norm[i_layer], w.ffn_w1[i_layer],
           w.ffn_w1_s[i_layer]);
    Matmul(ctx.ffn_w3x[i_layer], ctx.ffn_norm[i_layer], w.ffn_w3[i_layer],
           w.ffn_w3_s[i_layer]);
    SiLU(ctx.ffn_act[i_layer], ctx.ffn_w1x[i_layer]);
    Mul(ctx.ffn_dot[i_layer], ctx.ffn_act[i_layer], ctx.ffn_w3x[i_layer]);
    Matmul(ctx.ffn_out[i_layer], ctx.ffn_dot[i_layer], w.ffn_w2[i_layer],
           w.ffn_w2_s[i_layer]);
    Add(ctx.ffn_res[i_layer], ctx.attn_res[i_layer], ctx.ffn_out[i_layer]);
  }

  (void)ctx_logits;
  RMSNorm(ctx_final_norm, ctx.ffn_res[kNumLayers - 1], w.rms_final);
#endif
}

} // namespace llama2

#endif // BUILD_DECODE_KERNEL
