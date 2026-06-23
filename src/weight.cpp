#include "weight.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace llama2 {

struct QuantizedHeader {
  int dim;
  int hidden_dim;
  int num_layers;
  int num_heads;
  int num_kv_heads;
  int vocab_size;
  int seq_len;
  uint8_t shared_classifier;
  int group_size;
};

// Implement for initializing the tensor from the file.
// Template specialization for 1D tensor.
template <typename T, size_t Size>
void InitTensor(T (&tensor)[Size], std::ifstream& fs) {
  fs.read(reinterpret_cast<char*>(tensor), Size * sizeof(T));
}

// Implement for initializing the tensor from the file.
// Template specialization for 2D tensor.
template <typename T, size_t Outer, size_t Inner>
void InitTensor(T (&tensor)[Outer][Inner], std::ifstream& fs) {
  for (size_t outer = 0; outer < Outer; outer++) {
    InitTensor(tensor[outer], fs);
  }
}

// Implement for initializing the tensor from the file.
// Template specialization for 3D tensor.
template <typename T, size_t Outer, size_t Middle, size_t Inner>
void InitTensor(T (&tensor)[Outer][Middle][Inner], std::ifstream& fs) {
  for (size_t outer = 0; outer < Outer; outer++) {
    InitTensor(tensor[outer], fs);
  }
}

template <size_t Rows, size_t Cols, size_t Groups>
void InitQuantizedTensor(int8_t (&q)[Rows][Cols], float (&s)[Rows][Groups],
                         std::ifstream& fs) {
  fs.read(reinterpret_cast<char*>(q), Rows * Cols * sizeof(int8_t));
  fs.read(reinterpret_cast<char*>(s), Rows * Groups * sizeof(float));
}

template <size_t Layers, size_t Rows, size_t Cols, size_t Groups>
void InitQuantizedTensor(int8_t (&q)[Layers][Rows][Cols],
                         float (&s)[Layers][Rows][Groups],
                         std::ifstream& fs) {
  for (size_t layer = 0; layer < Layers; ++layer) {
    InitQuantizedTensor(q[layer], s[layer], fs);
  }
}

void InitTokenEmbedding(Tensor2dTok& tok, Tensor2dTokQ& q, Tensor2dTokS& s) {
  for (int row = 0; row < kVocabSize; ++row) {
    for (int col = 0; col < kDim; ++col) {
      tok[row][col] = static_cast<float>(q[row][col]) *
                      s[row][col / kQuantGroupSize];
    }
  }
}

void InitRoPETable(Tensor2dSinCos& cos_table, Tensor2dSinCos& sin_table) {
  constexpr int head_size = kDim / kNumHeads;
  for (int pos = 0; pos < kSeqLen; ++pos) {
    for (int i = 0; i < kSinCosTable; ++i) {
      const float freq =
          1.0f / std::pow(10000.0f, (2.0f * i) / static_cast<float>(head_size));
      const float val = static_cast<float>(pos) * freq;
      cos_table[pos][i] = std::cos(val);
      sin_table[pos][i] = std::sin(val);
    }
  }
}

QuantizedHeader ReadQuantizedHeader(std::ifstream& fs) {
  uint32_t magic = 0;
  int version = 0;
  QuantizedHeader h{};

  fs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  fs.read(reinterpret_cast<char*>(&version), sizeof(version));
  fs.read(reinterpret_cast<char*>(&h.dim), sizeof(int));
  fs.read(reinterpret_cast<char*>(&h.hidden_dim), sizeof(int));
  fs.read(reinterpret_cast<char*>(&h.num_layers), sizeof(int));
  fs.read(reinterpret_cast<char*>(&h.num_heads), sizeof(int));
  fs.read(reinterpret_cast<char*>(&h.num_kv_heads), sizeof(int));
  fs.read(reinterpret_cast<char*>(&h.vocab_size), sizeof(int));
  fs.read(reinterpret_cast<char*>(&h.seq_len), sizeof(int));
  fs.read(reinterpret_cast<char*>(&h.shared_classifier), sizeof(uint8_t));
  fs.read(reinterpret_cast<char*>(&h.group_size), sizeof(int));

  if (magic != 0x616b3432 || version != 2) {
    throw std::runtime_error("expected llama2.c version-2 q8 checkpoint");
  }
  if (h.dim != kDim || h.hidden_dim != kFFNDim ||
      h.num_layers != kNumLayers || h.num_heads != kNumHeads ||
      h.num_kv_heads != kNumKVHeads || h.vocab_size != kVocabSize ||
      h.seq_len != kSeqLen || h.group_size != kQuantGroupSize ||
      h.shared_classifier == 0) {
    throw std::runtime_error("q8 checkpoint shape does not match llama2.hls");
  }

  fs.seekg(256, std::ios::beg);
  return h;
}

// Implement for initializing tensors from a llama2.c version-2 q8 file.
void LoadWeights(Weights& w, Tensor2dTok& tok_emb_table, std::ifstream& fs) {
  ReadQuantizedHeader(fs);
  InitTensor(w.rms_att_w, fs);       // [kNumLayers, kDim]
  InitTensor(w.rms_ffn_w, fs);       // [kNumLayers, kDim]
  InitTensor(w.rms_final, fs);       // [kDim]
  InitQuantizedTensor(w.tok_emb_q, w.tok_emb_s, fs);
  InitTokenEmbedding(tok_emb_table, w.tok_emb_q, w.tok_emb_s);
  InitQuantizedTensor(w.attn_wq, w.attn_wq_s, fs);
  InitQuantizedTensor(w.attn_wk, w.attn_wk_s, fs);
  InitQuantizedTensor(w.attn_wv, w.attn_wv_s, fs);
  InitQuantizedTensor(w.attn_wo, w.attn_wo_s, fs);
  InitQuantizedTensor(w.ffn_w1, w.ffn_w1_s, fs);
  InitQuantizedTensor(w.ffn_w2, w.ffn_w2_s, fs);
  InitQuantizedTensor(w.ffn_w3, w.ffn_w3_s, fs);
  InitRoPETable(w.cos_table, w.sin_table);
}

} // namespace llama2
