#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include <string>
#include <stdint.h>

namespace llama2 {

constexpr int kDim = 288;
constexpr int kVocabSize = 32000;
constexpr int kNumLayers = 6;
constexpr int kNumHeads = 6;
constexpr int kNumKVHeads = 6;
constexpr int kSinCosTable = 24;
constexpr int kSeqLen = 256;
constexpr int kFFNDim = 768;
constexpr int kHalvedHeadDim = (kDim / kNumLayers);
constexpr int kQuantGroupSize = 32;
constexpr int kDimGroups = kDim / kQuantGroupSize;
constexpr int kFFNGroups = kFFNDim / kQuantGroupSize;

using Tensor1d = float[kDim];
using Tensor2dTok = float[kVocabSize][kDim];
using Tensor2dTokQ = int8_t[kVocabSize][kDim];
using Tensor2dTokS = float[kVocabSize][kDimGroups];
using Tensor2dAttn = float[kDim][kDim];
using Tensor3dAttn = float[kNumLayers][kDim][kDim];
using Tensor2dAttnQ = int8_t[kDim][kDim];
using Tensor3dAttnQ = int8_t[kNumLayers][kDim][kDim];
using Tensor2dAttnS = float[kDim][kDimGroups];
using Tensor3dAttnS = float[kNumLayers][kDim][kDimGroups];
using Tensor2dRMS = float[kNumLayers][kDim];
using Tensor1dSinCos = float[kSinCosTable];
using Tensor2dSinCos = float[kSeqLen][kSinCosTable];
using Tensor2dFFNA = float[kFFNDim][kDim];
using Tensor3dFFNA = float[kNumLayers][kFFNDim][kDim];
using Tensor2dFFNAQ = int8_t[kFFNDim][kDim];
using Tensor3dFFNAQ = int8_t[kNumLayers][kFFNDim][kDim];
using Tensor2dFFNAS = float[kFFNDim][kDimGroups];
using Tensor3dFFNAS = float[kNumLayers][kFFNDim][kDimGroups];
using Tensor1dFFNB = float[kFFNDim];
using Tensor2dFFNB = float[kDim][kFFNDim];
using Tensor3dFFNB = float[kNumLayers][kDim][kFFNDim];
using Tensor2dFFNBQ = int8_t[kDim][kFFNDim];
using Tensor3dFFNBQ = int8_t[kNumLayers][kDim][kFFNDim];
using Tensor2dFFNBS = float[kDim][kFFNGroups];
using Tensor3dFFNBS = float[kNumLayers][kDim][kFFNGroups];
using Tensor1dQKSM = float[kSeqLen];
using Tensor2dQKSM = float[kNumLayers][kSeqLen];
using Tensor2dFFNC = float[kNumLayers][kFFNDim];
using Tensor2dCache = float[kSeqLen][kDim];
using Tensor3dCache = float[kNumLayers][kSeqLen][kDim];
using Tensor1dLogits = float[kVocabSize];
using Tensor2d = float[kDim][kDim];
using Tensor3d = float[kDim][kDim][kDim];

void CopyTensor1d(Tensor1d& dst, const Tensor1d& src);
void CopyTensor2d(Tensor2d& dst, const Tensor2d& src);
void CopyTensor3d(Tensor3d& dst, const Tensor3d& src);

void Add(Tensor1d& out, const Tensor1d& in, float a);
void Mul(Tensor1dQKSM& out, const Tensor1dQKSM& in, float a);
void Sub(Tensor1d& out, const Tensor1d& in, float a);
void Div(Tensor1d& out, const Tensor1d& in, float a);

void Add(Tensor1d& out, const Tensor1d& lhs, const Tensor1d& rhs);
void Mul(Tensor1dFFNB& out, const Tensor1dFFNB& lhs, const Tensor1dFFNB& rhs);
void Sub(Tensor1d& out, const Tensor1d& lhs, const Tensor1d& rhs);
void Div(Tensor1d& out, const Tensor1d& lhs, const Tensor1d& rhs);

float InnerProduct(const Tensor1d& lhs, const Tensor1d& rhs);
void Matmul(Tensor1d& out, const Tensor1d& in, const Tensor2dAttnQ& w,
            const Tensor2dAttnS& s);
void MutmulRanged(Tensor1dQKSM& out, const Tensor1d& in, const Tensor2dCache& w,
                  int i_begin, int i_end, int j_begin, int j_end);
void MutmulRangedTranspose(Tensor1d& out, const Tensor1dQKSM& in,
                           const Tensor2dCache& w, int i_begin, int i_end,
                           int j_begin, int j_end);
void Matmul(Tensor1dFFNB& out, const Tensor1d& in, const Tensor2dFFNAQ& w,
            const Tensor2dFFNAS& s);
void Matmul(Tensor1d& out, const Tensor1dFFNB& in, const Tensor2dFFNBQ& w,
            const Tensor2dFFNBS& s);
void MutmulVocab(Tensor1dLogits& out, const Tensor1d& in,
                 const Tensor2dTokQ& w, const Tensor2dTokS& s);

void ReLU(Tensor1d& out, const Tensor1d& in);
void SiLU(Tensor1dFFNB& out, const Tensor1dFFNB& in);

void RMSNorm(Tensor1d& out, const Tensor1d& in, const Tensor1d& w);
void Softmax(Tensor1dQKSM& out, const Tensor1dQKSM& in, int max_pos = -1);
void Softmax(Tensor1dLogits& out, const Tensor1dLogits& in, int max_pos = -1);

std::pair<int, float> FindMaxIndexAndValue(const Tensor1d& in);
float Max(const Tensor1d& in);
int Argmax(const Tensor1dLogits& values);

void RoPE(Tensor1d& q_out, Tensor1d& k_out, const Tensor1d& q_in,
          const Tensor1d& k_in, const Tensor1dSinCos& cos_vec,
          const Tensor1dSinCos& sin_vec, int head_begin, int head_size);

} // namespace llama2

#endif // TENSOR_HPP_
