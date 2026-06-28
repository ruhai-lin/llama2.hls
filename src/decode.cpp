#include "decode.hpp"

#include <cmath>
#include <iostream>

namespace llama2 {

// Generate text from the model.
// This function is executed on the FPGA.
void Decode(int tok, // new token
            int pos, // new token position
            const Tensor1d& ctx_input, Tensor3dCache& ctx_k_cache,
            Tensor3dCache& ctx_v_cache, Tensor1d& ctx_final_norm,
            const Weights& w
#ifndef USE_CPU_ONLY
            ,
            FPGA& fpga
#endif // USE_CPU_ONLY
) {

  static Context ctx;

  float norm = 1 / std::sqrt(kDim / kNHeads); // 1/√d for sm(QK/√d)V

  // Embedding
  Tensor1d attn_input;
  for (int i_layer = 0; i_layer < kNLayers; ++i_layer) {

    if (i_layer == 0) {
      for (int idx = 0; idx < kDim; idx++) {
        attn_input[idx] = ctx_input[idx];
      }
    } else {
      for (int idx = 0; idx < kDim; idx++) {
        attn_input[idx] = ctx.ffn_res[i_layer - 1][idx];
      }
    }

    // -- Attention --

    // 1. RMS Normalize
#ifndef USE_CPU_ONLY
    RMSNormFPGA(ctx.attn_norm[i_layer], attn_input,
                fpga.weights.rms_att_w[i_layer], fpga);
#else
    RMSNorm(ctx.attn_norm[i_layer], attn_input, w.rms_att_w[i_layer]);
#endif

    // 2. Weight Multiple
#ifndef USE_CPU_ONLY
    MatmulFPGA(ctx.attn_wqx[i_layer], ctx.attn_norm[i_layer],
               fpga.weights.attn_wq[i_layer], fpga);
    MatmulFPGA(ctx.attn_wkx[i_layer], ctx.attn_norm[i_layer],
               fpga.weights.attn_wk[i_layer], fpga);
    MatmulFPGA(ctx.attn_wvx[i_layer], ctx.attn_norm[i_layer],
               fpga.weights.attn_wv[i_layer], fpga);
#else
    Matmul(ctx.attn_wqx[i_layer], ctx.attn_norm[i_layer], w.attn_wq[i_layer]);
    Matmul(ctx.attn_wkx[i_layer], ctx.attn_norm[i_layer], w.attn_wk[i_layer]);
    Matmul(ctx.attn_wvx[i_layer], ctx.attn_norm[i_layer], w.attn_wv[i_layer]);
#endif

    // 3. RoPE for each head
    for (int head = 0; head < kNHeads; ++head) {
#ifndef USE_CPU_ONLY
      RoPEFPGA(ctx.attn_q_r[i_layer], ctx.attn_k_r[i_layer],
               ctx.attn_wqx[i_layer], ctx.attn_wkx[i_layer], w.cos_table[pos],
               w.sin_table[pos], head * (kDim / kNHeads), fpga);
#else
      RoPE(ctx.attn_q_r[i_layer], ctx.attn_k_r[i_layer], ctx.attn_wqx[i_layer],
           ctx.attn_wkx[i_layer], w.cos_table[pos], w.sin_table[pos],
           head * (kDim / kNHeads));
#endif
    }

    // 4. Key / Value Cache
    CopyTensor1d(ctx_k_cache[i_layer][pos], ctx.attn_k_r[i_layer]);
    CopyTensor1d(ctx_v_cache[i_layer][pos], ctx.attn_wvx[i_layer]);

    // 5. Multi-Head Attention
    for (int i_head = 0; i_head < kNHeads; ++i_head) {

      int head_begin = i_head * (kDim / kNHeads);
      int head_end = (i_head + 1) * (kDim / kNHeads);

      // 5-1. QK
      MutmulRanged(ctx.attn_qk[i_layer], ctx.attn_q_r[i_layer],
                   ctx_k_cache[i_layer], 0, pos, head_begin, head_end);

      // 5-2. QK * 1/√d
#ifndef USE_CPU_ONLY
      MulFPGA(ctx.attn_qk[i_layer], ctx.attn_qk[i_layer], norm, fpga);
#else
      Mul(ctx.attn_qk[i_layer], ctx.attn_qk[i_layer], norm);
#endif

      // 5-3. Softmax( QK/√d )
#ifndef USE_CPU_ONLY
      SoftmaxFPGA(ctx.attn_sm[i_layer], ctx.attn_qk[i_layer], pos + 1, fpga);
#else
      Softmax(ctx.attn_sm[i_layer], ctx.attn_qk[i_layer], pos + 1);
#endif

      // 5-4. Softmax(QK/√d) . V
      MutmulRangedTranspose(ctx.attn_val[i_layer], ctx.attn_sm[i_layer],
                            ctx_v_cache[i_layer], head_begin, head_end, 0,
                            pos + 1);
    }

    // 6. Output (Merge Heads)
#ifndef USE_CPU_ONLY
    MatmulFPGA(ctx.attn_out[i_layer], ctx.attn_val[i_layer],
               fpga.weights.attn_wo[i_layer], fpga);
#else
    Matmul(ctx.attn_out[i_layer], ctx.attn_val[i_layer], w.attn_wo[i_layer]);
#endif

    // 7. Res connect
#ifndef USE_CPU_ONLY
    AddFPGA(ctx.attn_res[i_layer], attn_input, ctx.attn_out[i_layer], fpga);
#else
    Add(ctx.attn_res[i_layer], attn_input, ctx.attn_out[i_layer]);
#endif

    // -- FFN --

    // 1. RMS Normalize
#ifndef USE_CPU_ONLY
    RMSNormFPGA(ctx.ffn_norm[i_layer], ctx.attn_res[i_layer],
                fpga.weights.rms_ffn_w[i_layer], fpga);
#else
    RMSNorm(ctx.ffn_norm[i_layer], ctx.attn_res[i_layer], w.rms_ffn_w[i_layer]);
#endif

    // 2. w1 . x
#ifndef USE_CPU_ONLY
    MatmulFPGA(ctx.ffn_w1x[i_layer], ctx.ffn_norm[i_layer],
               fpga.weights.ffn_w1[i_layer], fpga);
#else
    Matmul(ctx.ffn_w1x[i_layer], ctx.ffn_norm[i_layer], w.ffn_w1[i_layer]);
#endif

    // 3. w3 . x
#ifndef USE_CPU_ONLY
    MatmulFPGA(ctx.ffn_w3x[i_layer], ctx.ffn_norm[i_layer],
               fpga.weights.ffn_w3[i_layer], fpga);
#else
    Matmul(ctx.ffn_w3x[i_layer], ctx.ffn_norm[i_layer], w.ffn_w3[i_layer]);
#endif

    // 4. SiLU( w1x )
    SiLU(ctx.ffn_act[i_layer], ctx.ffn_w1x[i_layer]);

    // 5. SiLU(w1x) * w3x
#ifndef USE_CPU_ONLY
    MulFPGA(ctx.ffn_dot[i_layer], ctx.ffn_act[i_layer], ctx.ffn_w3x[i_layer],
            fpga);
#else
    Mul(ctx.ffn_dot[i_layer], ctx.ffn_act[i_layer], ctx.ffn_w3x[i_layer]);
#endif

    // 6. w2 . SiLU(w1x)*w3x
#ifndef USE_CPU_ONLY
    MatmulFPGA(ctx.ffn_out[i_layer], ctx.ffn_dot[i_layer],
               fpga.weights.ffn_w2[i_layer], fpga);
#else
    Matmul(ctx.ffn_out[i_layer], ctx.ffn_dot[i_layer], w.ffn_w2[i_layer]);
#endif

    // 7. Res connect
#ifndef USE_CPU_ONLY
    AddFPGA(ctx.ffn_res[i_layer], ctx.attn_res[i_layer], ctx.ffn_out[i_layer],
            fpga);
#else
    Add(ctx.ffn_res[i_layer], ctx.attn_res[i_layer], ctx.ffn_out[i_layer]);
#endif
  }

  // -- Final RMS Normalize --
#ifndef USE_CPU_ONLY
  RMSNormFPGA(ctx_final_norm, ctx.ffn_res[kNLayers - 1],
              fpga.weights.rms_final, fpga);
#else
  RMSNorm(ctx_final_norm, ctx.ffn_res[kNLayers - 1], w.rms_final);
#endif

  return;
}

} // namespace llama2
