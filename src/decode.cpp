#ifdef BUILD_DECODE_KERNEL

#include "tensor.hpp"

#include <float.h>
#include <math.h>
#include <stdint.h>

namespace llama2 {
namespace {

constexpr int kHeadDim = kDim / kNumHeads;
constexpr int kMacLanes = 16;
constexpr int kAccStages = 8;
typedef float WeightWord __attribute__((vector_size(64)));

inline int rms_idx(int layer, int i) { return layer * kDim + i; }
inline int attn_idx(int layer, int row, int col) {
  return (layer * kDim + row) * kDim + col;
}
inline int ffn_a_idx(int layer, int row, int col) {
  return (layer * kFFNDim + row) * kDim + col;
}
inline int ffn_b_idx(int layer, int row, int col) {
  return (layer * kDim + row) * kFFNDim + col;
}
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

static inline float reduce16(const float acc[kMacLanes]) {
  const float sum01 = fadd(acc[0], acc[1]);
  const float sum23 = fadd(acc[2], acc[3]);
  const float sum45 = fadd(acc[4], acc[5]);
  const float sum67 = fadd(acc[6], acc[7]);
  const float sum89 = fadd(acc[8], acc[9]);
  const float sumab = fadd(acc[10], acc[11]);
  const float sumcd = fadd(acc[12], acc[13]);
  const float sumef = fadd(acc[14], acc[15]);
  const float sum03 = fadd(sum01, sum23);
  const float sum47 = fadd(sum45, sum67);
  const float sum8b = fadd(sum89, sumab);
  const float sumcf = fadd(sumcd, sumef);
  return fadd(fadd(sum03, sum47), fadd(sum8b, sumcf));
}

static inline float reduce_acc(const float acc[kMacLanes][kAccStages]) {
  float lanes[kMacLanes];
#pragma HLS ARRAY_PARTITION variable = lanes complete dim = 1

  for (int lane = 0; lane < kMacLanes; ++lane) {
#pragma HLS UNROLL
    lanes[lane] = acc[lane][0];
  }

  for (int stage = 1; stage < kAccStages; ++stage) {
    for (int lane = 0; lane < kMacLanes; ++lane) {
#pragma HLS UNROLL
      lanes[lane] = fadd(lanes[lane], acc[lane][stage]);
    }
  }

  return reduce16(lanes);
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

template <int OutSize, int InSize>
static void matmul(float (&out)[OutSize], const float (&in)[InSize],
                   const float* weight) {
#pragma HLS INLINE off
  float in_cache[InSize];
  float acc[kMacLanes][kAccStages];

#pragma HLS ARRAY_PARTITION variable = in_cache cyclic factor = 16 dim = 1
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0

  for (int col = 0; col < InSize; ++col) {
#pragma HLS PIPELINE II = 1
    in_cache[col] = in[col];
  }

  const WeightWord* weight_words =
      reinterpret_cast<const WeightWord*>(weight);

  for (int row = 0; row < OutSize; ++row) {
    for (int lane = 0; lane < kMacLanes; ++lane) {
#pragma HLS UNROLL
      for (int stage = 0; stage < kAccStages; ++stage) {
#pragma HLS UNROLL
        acc[lane][stage] = 0.0f;
      }
    }

    for (int tile = 0; tile < InSize / kMacLanes; ++tile) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter false
      const int base = tile * kMacLanes;
      const int stage = tile & (kAccStages - 1);
      const int weight_word_idx = (row * InSize + base) / kMacLanes;
      const WeightWord weight_vec = weight_words[weight_word_idx];

      for (int lane = 0; lane < kMacLanes; ++lane) {
#pragma HLS UNROLL
        acc[lane][stage] =
            fadd(acc[lane][stage], weight_vec[lane] * in_cache[base + lane]);
      }
    }

    out[row] = reduce_acc(acc);
  }
}

static void attn(Tensor1d& hidden, int layer, int pos,
                 const float* rms_att_w, const float* attn_wq,
                 const float* attn_wk, const float* attn_wv,
                 const float* attn_wo, const float* cos_table,
                 const float* sin_table, float* k_cache, float* v_cache) {
#pragma HLS INLINE off

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

  for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
    attn_input[i] = hidden[i];
  }

  rmsnorm(attn_norm, attn_input, &rms_att_w[rms_idx(layer, 0)]);

  matmul<kDim, kDim>(q, attn_norm, &attn_wq[attn_idx(layer, 0, 0)]);

  matmul<kDim, kDim>(k, attn_norm, &attn_wk[attn_idx(layer, 0, 0)]);

  matmul<kDim, kDim>(v, attn_norm, &attn_wv[attn_idx(layer, 0, 0)]);

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

  matmul<kDim, kDim>(attn_out, attn_val, &attn_wo[attn_idx(layer, 0, 0)]);

  add(hidden, attn_input, attn_out);
}

static void ffn(Tensor1d& hidden, int layer, const float* rms_ffn_w,
                const float* ffn_w1, const float* ffn_w2,
                const float* ffn_w3) {
#pragma HLS INLINE off

  Tensor1d ffn_norm;
  Tensor1dFFNB w1;
  Tensor1dFFNB w3;
  Tensor1dFFNB dot;
  Tensor1d out;

  rmsnorm(ffn_norm, hidden, &rms_ffn_w[rms_idx(layer, 0)]);

  matmul<kFFNDim, kDim>(w1, ffn_norm, &ffn_w1[ffn_a_idx(layer, 0, 0)]);

  matmul<kFFNDim, kDim>(w3, ffn_norm, &ffn_w3[ffn_a_idx(layer, 0, 0)]);

  silu_mul(dot, w1, w3);

  matmul<kDim, kFFNDim>(out, dot, &ffn_w2[ffn_b_idx(layer, 0, 0)]);

  add_inplace(hidden, out);
}

static uint32_t lm_head(const Tensor1d& hidden, const float* tok_emb_table,
                        const float* rms_final) {
#pragma HLS INLINE off

  Tensor1d final_norm;
  float in_cache[kDim];
  float acc[kMacLanes][kAccStages];

#pragma HLS ARRAY_PARTITION variable = in_cache cyclic factor = 16 dim = 1
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0

  rmsnorm(final_norm, hidden, rms_final);

  for (int col = 0; col < kDim; ++col) {
#pragma HLS PIPELINE II = 1
    in_cache[col] = final_norm[col];
  }

  const WeightWord* weight_words =
      reinterpret_cast<const WeightWord*>(tok_emb_table);

  float best_score = -FLT_MAX;
  uint32_t best_token = 0;
  for (int row = 0; row < kVocabSize; ++row) {
    for (int lane = 0; lane < kMacLanes; ++lane) {
#pragma HLS UNROLL
      for (int stage = 0; stage < kAccStages; ++stage) {
#pragma HLS UNROLL
        acc[lane][stage] = 0.0f;
      }
    }

    for (int tile = 0; tile < kDim / kMacLanes; ++tile) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter false
      const int base = tile * kMacLanes;
      const int stage = tile & (kAccStages - 1);
      const int weight_word_idx = (row * kDim + base) / kMacLanes;
      const WeightWord weight_vec = weight_words[weight_word_idx];

      for (int lane = 0; lane < kMacLanes; ++lane) {
#pragma HLS UNROLL
        acc[lane][stage] =
            fadd(acc[lane][stage], weight_vec[lane] * in_cache[base + lane]);
      }
    }

    const float score = reduce_acc(acc);
    if (score > best_score) {
      best_score = score;
      best_token = static_cast<uint32_t>(row);
    }
  }
  return best_token;
}

} // namespace
} // namespace llama2

using namespace llama2;

extern "C" {

void decode(int token, int pos, const float* tok_emb_table,
            const float* rms_att_w, const float* attn_wq,
            const float* attn_wk, const float* attn_wv, const float* attn_wo,
            const float* rms_ffn_w, const float* ffn_w1,
            const float* ffn_w2, const float* ffn_w3, const float* rms_final,
            const float* cos_table, const float* sin_table, float* k_cache,
            float* v_cache, uint32_t* next_token) {
// -------------------------
// Stage 0: AXI interfaces.
// -------------------------
#pragma HLS INTERFACE m_axi port = tok_emb_table bundle = gmem
#pragma HLS INTERFACE m_axi port = rms_att_w bundle = gmem
#pragma HLS INTERFACE m_axi port = attn_wq bundle = gmem
#pragma HLS INTERFACE m_axi port = attn_wk bundle = gmem
#pragma HLS INTERFACE m_axi port = attn_wv bundle = gmem
#pragma HLS INTERFACE m_axi port = attn_wo bundle = gmem
#pragma HLS INTERFACE m_axi port = rms_ffn_w bundle = gmem
#pragma HLS INTERFACE m_axi port = ffn_w1 bundle = gmem
#pragma HLS INTERFACE m_axi port = ffn_w2 bundle = gmem
#pragma HLS INTERFACE m_axi port = ffn_w3 bundle = gmem
#pragma HLS INTERFACE m_axi port = rms_final bundle = gmem
#pragma HLS INTERFACE m_axi port = cos_table bundle = gmem
#pragma HLS INTERFACE m_axi port = sin_table bundle = gmem
#pragma HLS INTERFACE m_axi port = k_cache bundle = gmem
#pragma HLS INTERFACE m_axi port = v_cache bundle = gmem
#pragma HLS INTERFACE m_axi port = next_token bundle = gmem

  Tensor1d hidden;
  load_token(hidden, token, tok_emb_table);

  for (int layer = 0; layer < kNumLayers; ++layer) {
    attn(hidden, layer, pos, rms_att_w, attn_wq, attn_wk, attn_wv, attn_wo,
         cos_table, sin_table, k_cache, v_cache);
    ffn(hidden, layer, rms_ffn_w, ffn_w1, ffn_w2, ffn_w3);
  }

  *next_token = lm_head(hidden, tok_emb_table, rms_final);
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
  (void)next_token;
  static Context ctx;

  const int head_dim = kDim / kNumLayers;
  float norm = 1 / std::sqrt(head_dim);

  Tensor1d attn_input;
  for (int i_layer = 0; i_layer < kNumLayers; ++i_layer) {
    if (i_layer == 0) {
      CopyTensor1d(attn_input, ctx_input);
    } else {
      CopyTensor1d(attn_input, ctx.ffn_res[i_layer - 1]);
    }

    RMSNorm(ctx.attn_norm[i_layer], attn_input, w.rms_att_w[i_layer]);
    Matmul(ctx.attn_wqx[i_layer], ctx.attn_norm[i_layer], w.attn_wq[i_layer]);
    Matmul(ctx.attn_wkx[i_layer], ctx.attn_norm[i_layer], w.attn_wk[i_layer]);
    Matmul(ctx.attn_wvx[i_layer], ctx.attn_norm[i_layer], w.attn_wv[i_layer]);

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

    Matmul(ctx.attn_out[i_layer], ctx.attn_val[i_layer], w.attn_wo[i_layer]);
    Add(ctx.attn_res[i_layer], attn_input, ctx.attn_out[i_layer]);
    RMSNorm(ctx.ffn_norm[i_layer], ctx.attn_res[i_layer], w.rms_ffn_w[i_layer]);
    Matmul(ctx.ffn_w1x[i_layer], ctx.ffn_norm[i_layer], w.ffn_w1[i_layer]);
    Matmul(ctx.ffn_w3x[i_layer], ctx.ffn_norm[i_layer], w.ffn_w3[i_layer]);
    SiLU(ctx.ffn_act[i_layer], ctx.ffn_w1x[i_layer]);
    Mul(ctx.ffn_dot[i_layer], ctx.ffn_act[i_layer], ctx.ffn_w3x[i_layer]);
    Matmul(ctx.ffn_out[i_layer], ctx.ffn_dot[i_layer], w.ffn_w2[i_layer]);
    Add(ctx.ffn_res[i_layer], ctx.attn_res[i_layer], ctx.ffn_out[i_layer]);
  }

  (void)ctx_logits;
  RMSNorm(ctx_final_norm, ctx.ffn_res[kNumLayers - 1], w.rms_final);
#endif
}

} // namespace llama2

#endif // BUILD_DECODE_KERNEL
