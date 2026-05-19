#ifdef BUILD_DECODE_KERNEL

#include "tensor.hpp"

#include "kernel_add.cpp"
#include "kernel_matmul.cpp"
#include "kernel_mul.cpp"
#include "kernel_rmsnorm.cpp"
#include "kernel_rope.cpp"
#include "kernel_softmax.cpp"

#include <math.h>

namespace swan {

constexpr int kHeadDim = kDim / kNumHeads;

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

} // namespace swan

using namespace swan;

extern "C" {

void kernel_decode(int token, int pos, const float* tok_emb_table,
                   const float* rms_att_w, const float* attn_wq,
                   const float* attn_wk, const float* attn_wv,
                   const float* attn_wo, const float* rms_ffn_w,
                   const float* ffn_w1, const float* ffn_w2,
                   const float* ffn_w3, const float* rms_final,
                   const float* cos_table, const float* sin_table,
                   float* k_cache, float* v_cache, float* logits) {
#pragma HLS INTERFACE m_axi port = tok_emb_table bundle = gmem0
#pragma HLS INTERFACE m_axi port = rms_att_w bundle = gmem1
#pragma HLS INTERFACE m_axi port = attn_wq bundle = gmem2
#pragma HLS INTERFACE m_axi port = attn_wk bundle = gmem3
#pragma HLS INTERFACE m_axi port = attn_wv bundle = gmem4
#pragma HLS INTERFACE m_axi port = attn_wo bundle = gmem5
#pragma HLS INTERFACE m_axi port = rms_ffn_w bundle = gmem6
#pragma HLS INTERFACE m_axi port = ffn_w1 bundle = gmem7
#pragma HLS INTERFACE m_axi port = ffn_w2 bundle = gmem8
#pragma HLS INTERFACE m_axi port = ffn_w3 bundle = gmem9
#pragma HLS INTERFACE m_axi port = rms_final bundle = gmem10
#pragma HLS INTERFACE m_axi port = cos_table bundle = gmem11
#pragma HLS INTERFACE m_axi port = sin_table bundle = gmem12
#pragma HLS INTERFACE m_axi port = k_cache bundle = gmem13
#pragma HLS INTERFACE m_axi port = v_cache bundle = gmem14
#pragma HLS INTERFACE m_axi port = logits bundle = gmem15

  Tensor1d hidden;
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
  Tensor1d attn_res;
  Tensor1d ffn_norm;
  Tensor1dFFNB ffn_w1x;
  Tensor1dFFNB ffn_w3x;
  Tensor1dFFNB ffn_dot;
  Tensor1d ffn_out;
  Tensor1d final_norm;

  for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
    hidden[i] = tok_emb_table[token * kDim + i];
  }

  const float attn_scale = 1.0f / sqrtf(static_cast<float>(kHeadDim));

  for (int layer = 0; layer < kNumLayers; ++layer) {
    for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
      attn_input[i] = hidden[i];
    }

    kernel_rmsnorm(attn_norm, attn_input, &rms_att_w[rms_idx(layer, 0)]);
    kernel_matmul(q, kDim, attn_norm, kDim, &attn_wq[attn_idx(layer, 0, 0)]);
    kernel_matmul(k, kDim, attn_norm, kDim, &attn_wk[attn_idx(layer, 0, 0)]);
    kernel_matmul(v, kDim, attn_norm, kDim, &attn_wv[attn_idx(layer, 0, 0)]);
    kernel_rope(q_rot, k_rot, q, k, cos_table, sin_table, pos);

    for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
      k_cache[cache_idx(layer, pos, i)] = k_rot[i];
      v_cache[cache_idx(layer, pos, i)] = v[i];
      attn_val[i] = 0.0f;
    }

    for (int head = 0; head < kNumHeads; ++head) {
      const int head_begin = head * kHeadDim;
      const int head_end = head_begin + kHeadDim;

      for (int t = 0; t <= pos; ++t) {
        float sum = 0.0f;
        for (int i = head_begin; i < head_end; ++i) {
#pragma HLS PIPELINE II = 1
          sum += q_rot[i] * k_cache[cache_idx(layer, t, i)];
        }
        qk[t] = sum * attn_scale;
      }

      kernel_softmax(sm, qk, pos + 1);

      for (int i = head_begin; i < head_end; ++i) {
        float sum = 0.0f;
        for (int t = 0; t <= pos; ++t) {
#pragma HLS PIPELINE II = 1
          sum += sm[t] * v_cache[cache_idx(layer, t, i)];
        }
        attn_val[i] = sum;
      }
    }

    kernel_matmul(attn_out, kDim, attn_val, kDim,
                  &attn_wo[attn_idx(layer, 0, 0)]);
    kernel_add(attn_res, attn_input, attn_out);
    kernel_rmsnorm(ffn_norm, attn_res, &rms_ffn_w[rms_idx(layer, 0)]);
    kernel_matmul(ffn_w1x, kFFNDim, ffn_norm, kDim,
                  &ffn_w1[ffn_a_idx(layer, 0, 0)]);
    kernel_matmul(ffn_w3x, kFFNDim, ffn_norm, kDim,
                  &ffn_w3[ffn_a_idx(layer, 0, 0)]);
    kernel_mul(ffn_dot, ffn_w1x, ffn_w3x);
    kernel_matmul(ffn_out, kDim, ffn_dot, kFFNDim,
                  &ffn_w2[ffn_b_idx(layer, 0, 0)]);
    kernel_add(hidden, attn_res, ffn_out);
  }

  kernel_rmsnorm(final_norm, hidden, rms_final);

  for (int tok = 0; tok < kVocabSize; ++tok) {
    float sum = 0.0f;
    for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
      sum += tok_emb_table[tok * kDim + i] * final_norm[i];
    }
    logits[tok] = sum;
  }
}

} // extern "C"

#else

#include "decode.hpp"

#include <cmath>
#include <cstring>

namespace swan {

void Decode(int tok, int pos, const Tensor1d& ctx_input,
            Tensor3dCache& ctx_k_cache, Tensor3dCache& ctx_v_cache,
            Tensor1d& ctx_final_norm, Tensor1dLogits& ctx_logits,
            const Weights& w
#ifndef USE_CPU_ONLY
            ,
            cl::CommandQueue q, cl::Kernel kernel_decode, float* ptr_logits,
            cl::Buffer buffer_logits
#endif // USE_CPU_ONLY
) {
#ifndef USE_CPU_ONLY
  (void)ctx_input;
  (void)ctx_k_cache;
  (void)ctx_v_cache;
  (void)ctx_final_norm;
  (void)w;

  kernel_decode.setArg(0, tok);
  kernel_decode.setArg(1, pos);
  q.enqueueNDRangeKernel(kernel_decode, cl::NullRange, cl::NDRange(1),
                         cl::NullRange);
  q.enqueueMigrateMemObjects({buffer_logits}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  std::memcpy(ctx_logits, ptr_logits, sizeof(Tensor1dLogits));
#else
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

} // namespace swan

#endif // BUILD_DECODE_KERNEL
