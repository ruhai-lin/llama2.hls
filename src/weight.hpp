#ifndef WEIGHT_HPP_
#define WEIGHT_HPP_

#include <fstream>
#include <string>

#include "tensor.hpp"

namespace llama2 {

struct Weights {
  // Token embedding projection
  Tensor2dTokQ tok_emb_q; // [vocab_size, dim]
  Tensor2dTokS tok_emb_s; // [vocab_size, dim / group_size]

  // Attention
  Tensor2dRMS rms_att_w; // [n_layers, dim]
  Tensor3dAttnQ attn_wq; // [n_layers, dim, dim]
  Tensor3dAttnQ attn_wk; // [n_layers, dim, dim]
  Tensor3dAttnQ attn_wv; // [n_layers, dim, dim]
  Tensor3dAttnQ attn_wo; // [n_layers, dim, dim]
  Tensor3dAttnS attn_wq_s;
  Tensor3dAttnS attn_wk_s;
  Tensor3dAttnS attn_wv_s;
  Tensor3dAttnS attn_wo_s;

  // FFN
  Tensor2dRMS rms_ffn_w; // [n_layers, dim]
  Tensor3dFFNAQ ffn_w1;  // [n_layers, ffn_dim, dim]
  Tensor3dFFNBQ ffn_w2;  // [n_layers, dim, ffn_dim]
  Tensor3dFFNAQ ffn_w3;  // [n_layers, ffn_dim, dim]
  Tensor3dFFNAS ffn_w1_s;
  Tensor3dFFNBS ffn_w2_s;
  Tensor3dFFNAS ffn_w3_s;

  // Final rmsnorm
  Tensor1d rms_final; // [dim]

  // freq_cis for RoPE relatively positional embeddings
  Tensor2dSinCos cos_table; // [seq_len, (dim/n_heads)/2]
  Tensor2dSinCos sin_table; // [seq_len, (dim/n_heads)/2]
};

void LoadWeights(Weights& w, Tensor2dTok& tok, std::ifstream& fs);

} // namespace llama2

#endif // WEIGHT_HPP_
