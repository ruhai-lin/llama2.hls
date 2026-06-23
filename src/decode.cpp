#ifdef BUILD_DECODE_KERNEL

#include "tensor.hpp"

#include "kernel_add.cpp"
#include "kernel_matmul.cpp"
#include "kernel_mul.cpp"
#include "kernel_rmsnorm.cpp"
#include "kernel_rope.cpp"
#include "kernel_silu.cpp"
#include "kernel_softmax.cpp"

#include <float.h>
#include <math.h>
#include <stdint.h>

namespace llama2 {

constexpr int kHeadDim = kDim / kNumHeads;
constexpr int kVocabTile = 128;
constexpr int kTasksPerLayer = 7;

enum class Task {
  AttnQ = 0,
  AttnK = 1,
  AttnV = 2,
  AttnO = 3,
  FfnW1 = 4,
  FfnW3 = 5,
  FfnW2 = 6,
  LmHead = 7,
};

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
inline int attn_s_idx(int layer, int row, int group) {
  return (layer * kDim + row) * kDimGroups + group;
}
inline int ffn_a_s_idx(int layer, int row, int group) {
  return (layer * kFFNDim + row) * kDimGroups + group;
}
inline int ffn_b_s_idx(int layer, int row, int group) {
  return (layer * kDim + row) * kFFNGroups + group;
}
inline int cache_idx(int layer, int pos, int i) {
  return (layer * kSeqLen + pos) * kDim + i;
}

} // namespace llama2

using namespace llama2;

extern "C" {

void kernel_decode(int token, int pos, const float* tok_emb_table,
                   const int8_t* tok_emb_q, const float* tok_emb_s,
                   const float* rms_att_w, const int8_t* attn_wq,
                   const float* attn_wq_s, const int8_t* attn_wk,
                   const float* attn_wk_s, const int8_t* attn_wv,
                   const float* attn_wv_s, const int8_t* attn_wo,
                   const float* attn_wo_s, const float* rms_ffn_w,
                   const int8_t* ffn_w1, const float* ffn_w1_s,
                   const int8_t* ffn_w2, const float* ffn_w2_s,
                   const int8_t* ffn_w3, const float* ffn_w3_s,
                   const float* rms_final, const float* cos_table,
                   const float* sin_table, float* k_cache, float* v_cache,
                   uint32_t* next_token) {
// -------------------------
// Stage 0: AXI interfaces.
// -------------------------
#pragma HLS INTERFACE m_axi port = tok_emb_table bundle = gmem
#pragma HLS INTERFACE m_axi port = tok_emb_q bundle = gmem
#pragma HLS INTERFACE m_axi port = tok_emb_s bundle = gmem
#pragma HLS INTERFACE m_axi port = rms_att_w bundle = gmem
#pragma HLS INTERFACE m_axi port = attn_wq bundle = gmem
#pragma HLS INTERFACE m_axi port = attn_wq_s bundle = gmem
#pragma HLS INTERFACE m_axi port = attn_wk bundle = gmem
#pragma HLS INTERFACE m_axi port = attn_wk_s bundle = gmem
#pragma HLS INTERFACE m_axi port = attn_wv bundle = gmem
#pragma HLS INTERFACE m_axi port = attn_wv_s bundle = gmem
#pragma HLS INTERFACE m_axi port = attn_wo bundle = gmem
#pragma HLS INTERFACE m_axi port = attn_wo_s bundle = gmem
#pragma HLS INTERFACE m_axi port = rms_ffn_w bundle = gmem
#pragma HLS INTERFACE m_axi port = ffn_w1 bundle = gmem
#pragma HLS INTERFACE m_axi port = ffn_w1_s bundle = gmem
#pragma HLS INTERFACE m_axi port = ffn_w2 bundle = gmem
#pragma HLS INTERFACE m_axi port = ffn_w2_s bundle = gmem
#pragma HLS INTERFACE m_axi port = ffn_w3 bundle = gmem
#pragma HLS INTERFACE m_axi port = ffn_w3_s bundle = gmem
#pragma HLS INTERFACE m_axi port = rms_final bundle = gmem
#pragma HLS INTERFACE m_axi port = cos_table bundle = gmem
#pragma HLS INTERFACE m_axi port = sin_table bundle = gmem
#pragma HLS INTERFACE m_axi port = k_cache bundle = gmem
#pragma HLS INTERFACE m_axi port = v_cache bundle = gmem
#pragma HLS INTERFACE m_axi port = next_token bundle = gmem

  // -------------------------
  // Stage 1: on-chip buffers.
  // -------------------------
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
  Tensor1dFFNB ffn_act;
  Tensor1dFFNB ffn_dot;
  Tensor1d ffn_out;
  Tensor1d final_norm;
  float score_tile[kVocabTile];

  // -------------------------
  // Stage 2: token embedding.
  // -------------------------
  for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
    hidden[i] = tok_emb_table[token * kDim + i];
  }

  const float attn_scale = 1.0f / sqrtf(static_cast<float>(kHeadDim));
  constexpr int kLayerLinearTaskCount = kNumLayers * kTasksPerLayer;
  constexpr int kLmHeadTileCount =
      (kVocabSize + kVocabTile - 1) / kVocabTile;
  constexpr int kLinearTaskCount = kLayerLinearTaskCount + kLmHeadTileCount;
  float best_score = -FLT_MAX;
  uint32_t best_token = 0;

  // -------------------------
  // Stage 3: shared GEMV schedule.
  // -------------------------
  for (int linear_task = 0; linear_task < kLinearTaskCount; ++linear_task) {
    const bool lm_head_task = linear_task >= kLayerLinearTaskCount;
    const int layer = lm_head_task ? 0 : linear_task / kTasksPerLayer;
    const int layer_task =
        lm_head_task ? 0 : linear_task - layer * kTasksPerLayer;
    const int lm_head_tile = linear_task - kLayerLinearTaskCount;
    const int vocab_base = lm_head_tile * kVocabTile;
    const Task task =
        lm_head_task ? Task::LmHead : static_cast<Task>(layer_task);

    const float* linear_src = final_norm;
    const int8_t* linear_weight_q = tok_emb_q;
    const float* linear_weight_s = tok_emb_s;
    float* linear_dst = hidden;
    int linear_rows = kDim;
    int linear_cols = kDim;

    switch (task) {
    case Task::LmHead:
      if (lm_head_tile == 0) {
        kernel_rmsnorm(final_norm, hidden, rms_final);
        best_score = -FLT_MAX;
        best_token = 0;
      }
      linear_src = final_norm;
      linear_weight_q = &tok_emb_q[vocab_base * kDim];
      linear_weight_s = &tok_emb_s[vocab_base * kDimGroups];
      linear_dst = score_tile;
      linear_rows = kVocabTile;
      if (vocab_base + kVocabTile > kVocabSize) {
        linear_rows = kVocabSize - vocab_base;
      }
      linear_cols = kDim;
      break;
    case Task::AttnQ:
      for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
        attn_input[i] = hidden[i];
      }
      kernel_rmsnorm(attn_norm, attn_input, &rms_att_w[rms_idx(layer, 0)]);
      linear_src = attn_norm;
      linear_weight_q = &attn_wq[attn_idx(layer, 0, 0)];
      linear_weight_s = &attn_wq_s[attn_s_idx(layer, 0, 0)];
      linear_dst = q;
      linear_rows = kDim;
      linear_cols = kDim;
      break;
    case Task::AttnK:
      linear_src = attn_norm;
      linear_weight_q = &attn_wk[attn_idx(layer, 0, 0)];
      linear_weight_s = &attn_wk_s[attn_s_idx(layer, 0, 0)];
      linear_dst = k;
      linear_rows = kDim;
      linear_cols = kDim;
      break;
    case Task::AttnV:
      linear_src = attn_norm;
      linear_weight_q = &attn_wv[attn_idx(layer, 0, 0)];
      linear_weight_s = &attn_wv_s[attn_s_idx(layer, 0, 0)];
      linear_dst = v;
      linear_rows = kDim;
      linear_cols = kDim;
      break;
    case Task::AttnO:
      // Prepare attention values before the output projection.
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

      linear_src = attn_val;
      linear_weight_q = &attn_wo[attn_idx(layer, 0, 0)];
      linear_weight_s = &attn_wo_s[attn_s_idx(layer, 0, 0)];
      linear_dst = attn_out;
      linear_rows = kDim;
      linear_cols = kDim;
      break;
    case Task::FfnW1:
      kernel_add(attn_res, attn_input, attn_out);
      kernel_rmsnorm(ffn_norm, attn_res, &rms_ffn_w[rms_idx(layer, 0)]);
      linear_src = ffn_norm;
      linear_weight_q = &ffn_w1[ffn_a_idx(layer, 0, 0)];
      linear_weight_s = &ffn_w1_s[ffn_a_s_idx(layer, 0, 0)];
      linear_dst = ffn_w1x;
      linear_rows = kFFNDim;
      linear_cols = kDim;
      break;
    case Task::FfnW3:
      linear_src = ffn_norm;
      linear_weight_q = &ffn_w3[ffn_a_idx(layer, 0, 0)];
      linear_weight_s = &ffn_w3_s[ffn_a_s_idx(layer, 0, 0)];
      linear_dst = ffn_w3x;
      linear_rows = kFFNDim;
      linear_cols = kDim;
      break;
    case Task::FfnW2:
      kernel_silu(ffn_act, ffn_w1x);
      kernel_mul(ffn_dot, ffn_act, ffn_w3x);
      linear_src = ffn_dot;
      linear_weight_q = &ffn_w2[ffn_b_idx(layer, 0, 0)];
      linear_weight_s = &ffn_w2_s[ffn_b_s_idx(layer, 0, 0)];
      linear_dst = ffn_out;
      linear_rows = kDim;
      linear_cols = kFFNDim;
      break;
    }

    kernel_matmul(linear_dst, linear_rows, linear_src, linear_cols,
                  linear_weight_q, linear_weight_s);

    switch (task) {
    case Task::LmHead:
      for (int i = 0; i < linear_rows; ++i) {
#pragma HLS PIPELINE II = 1
        if (score_tile[i] > best_score) {
          best_score = score_tile[i];
          best_token = static_cast<uint32_t>(vocab_base + i);
        }
      }
      if (lm_head_tile == kLmHeadTileCount - 1) {
        *next_token = best_token;
      }
      break;
    case Task::FfnW2:
      kernel_add(hidden, attn_res, ffn_out);
      break;
    default:
      break;
    }
  }
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
            cl::CommandQueue q, cl::Kernel kernel_decode, uint32_t* ptr_next,
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

  kernel_decode.setArg(0, tok);
  kernel_decode.setArg(1, pos);
  q.enqueueTask(kernel_decode);
  q.enqueueMigrateMemObjects({buffer_next}, CL_MIGRATE_MEM_OBJECT_HOST);
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
