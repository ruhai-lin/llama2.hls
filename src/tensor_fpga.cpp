#include "tensor_fpga.hpp"

#ifndef USE_CPU_ONLY

#include <algorithm>

namespace llama2 {

namespace {

cl::Buffer UploadReadOnly(cl::Context context, cl::CommandQueue q,
                          const float* data, size_t count) {
  cl::Buffer buffer(context, CL_MEM_READ_ONLY, count * sizeof(float));
  q.enqueueWriteBuffer(buffer, CL_TRUE, 0, count * sizeof(float), data);
  return buffer;
}

void RunMatmul(float* out, const float* in, int in_size, int out_size,
               cl::Buffer weight_buffer, FPGA& fpga) {
  for (int i = 0; i < in_size; ++i) {
    fpga.ptr_a[i] = in[i];
  }

  fpga.q.enqueueMigrateMemObjects({fpga.buffer_a}, 0);
  fpga.kernel_matmul.setArg(1, weight_buffer);
  fpga.kernel_matmul.setArg(3, in_size);
  fpga.kernel_matmul.setArg(4, out_size);
  fpga.q.enqueueTask(fpga.kernel_matmul);
  fpga.q.enqueueMigrateMemObjects({fpga.buffer_result},
                                  CL_MIGRATE_MEM_OBJECT_HOST);
  fpga.q.finish();

  for (int i = 0; i < out_size; ++i) {
    out[i] = fpga.ptr_result[i];
  }
}

void RunRMSNorm(float* out, const float* in, cl::Buffer weight_buffer,
                FPGA& fpga) {
  for (int i = 0; i < kDim; ++i) {
    fpga.ptr_a[i] = in[i];
  }

  fpga.q.enqueueMigrateMemObjects({fpga.buffer_a}, 0);
  fpga.kernel_rmsnorm.setArg(1, weight_buffer);
  fpga.kernel_rmsnorm.setArg(3, kDim);
  fpga.q.enqueueTask(fpga.kernel_rmsnorm);
  fpga.q.enqueueMigrateMemObjects({fpga.buffer_result},
                                  CL_MIGRATE_MEM_OBJECT_HOST);
  fpga.q.finish();

  for (int i = 0; i < kDim; ++i) {
    out[i] = fpga.ptr_result[i];
  }
}

} // namespace

void UploadDeviceWeights(DeviceWeights& out, const Weights& w,
                         const Tensor2dTok& tok_emb_table, cl::Context context,
                         cl::CommandQueue q) {
  for (int layer = 0; layer < kNLayers; ++layer) {
    out.rms_att_w[layer] =
        UploadReadOnly(context, q, w.rms_att_w[layer], kDim);
    out.attn_wq[layer] =
        UploadReadOnly(context, q, w.attn_wq[layer][0], kDim * kDim);
    out.attn_wk[layer] =
        UploadReadOnly(context, q, w.attn_wk[layer][0], kDim * kDim);
    out.attn_wv[layer] =
        UploadReadOnly(context, q, w.attn_wv[layer][0], kDim * kDim);
    out.attn_wo[layer] =
        UploadReadOnly(context, q, w.attn_wo[layer][0], kDim * kDim);

    out.rms_ffn_w[layer] =
        UploadReadOnly(context, q, w.rms_ffn_w[layer], kDim);
    out.ffn_w1[layer] =
        UploadReadOnly(context, q, w.ffn_w1[layer][0], kHiddenDim * kDim);
    out.ffn_w2[layer] =
        UploadReadOnly(context, q, w.ffn_w2[layer][0], kDim * kHiddenDim);
    out.ffn_w3[layer] =
        UploadReadOnly(context, q, w.ffn_w3[layer][0], kHiddenDim * kDim);
  }

  out.rms_final = UploadReadOnly(context, q, w.rms_final, kDim);
  for (int tile = 0; tile < (kVocabSize + kVocabTile - 1) / kVocabTile;
       ++tile) {
    const int row_begin = tile * kVocabTile;
    out.lm_head_rows[tile] = std::min(kVocabTile, kVocabSize - row_begin);
    out.lm_head[tile] = UploadReadOnly(context, q, tok_emb_table[row_begin],
                                       out.lm_head_rows[tile] * kDim);
  }
}

/* ---------------------------------  /
      Basic Arithmetic Operations
/  --------------------------------- */

void AddFPGA(Tensor1d& out, const Tensor1d& lhs, const Tensor1d& rhs,
             FPGA& fpga) {
  for (int i = 0; i < kDim; i++) {
    fpga.ptr_a[i] = lhs[i];
  }
  for (int i = 0; i < kDim; i++) {
    fpga.ptr_b[i] = rhs[i];
  }
  fpga.q.enqueueMigrateMemObjects({fpga.buffer_a, fpga.buffer_b}, 0);
  fpga.kernel_add.setArg(3, kDim);
  fpga.q.enqueueTask(fpga.kernel_add);
  fpga.q.enqueueMigrateMemObjects({fpga.buffer_result},
                                  CL_MIGRATE_MEM_OBJECT_HOST);
  fpga.q.finish();
  for (int i = 0; i < kDim; i++) {
    out[i] = fpga.ptr_result[i];
  }
}

void MulFPGA(Tensor1dQKSM& out, const Tensor1dQKSM& in, float a, FPGA& fpga) {
  for (int i = 0; i < kSeqLen; i++) {
    fpga.ptr_a[i] = in[i];
  }
  for (int i = 0; i < kSeqLen; i++) {
    fpga.ptr_b[i] = a;
  }
  fpga.q.enqueueMigrateMemObjects({fpga.buffer_a, fpga.buffer_b}, 0);
  fpga.kernel_mul.setArg(3, kSeqLen);
  fpga.q.enqueueTask(fpga.kernel_mul);
  fpga.q.enqueueMigrateMemObjects({fpga.buffer_result},
                                  CL_MIGRATE_MEM_OBJECT_HOST);
  fpga.q.finish();
  for (int i = 0; i < kSeqLen; i++) {
    out[i] = fpga.ptr_result[i];
  }
}

void MulFPGA(Tensor1dFFNB& out, const Tensor1dFFNB& lhs,
             const Tensor1dFFNB& rhs, FPGA& fpga) {
  for (int i = 0; i < kHiddenDim; i++) {
    fpga.ptr_a[i] = lhs[i];
  }
  for (int i = 0; i < kHiddenDim; i++) {
    fpga.ptr_b[i] = rhs[i];
  }
  fpga.q.enqueueMigrateMemObjects({fpga.buffer_a, fpga.buffer_b}, 0);
  fpga.kernel_mul.setArg(3, kHiddenDim);
  fpga.q.enqueueTask(fpga.kernel_mul);
  fpga.q.enqueueMigrateMemObjects({fpga.buffer_result},
                                  CL_MIGRATE_MEM_OBJECT_HOST);
  fpga.q.finish();
  for (int i = 0; i < kHiddenDim; i++) {
    out[i] = fpga.ptr_result[i];
  }
}

/* ---------------------------------  /
           Matrix Operations
/  --------------------------------- */

void MatmulFPGA(Tensor1d& out, const Tensor1d& in, cl::Buffer weight_buffer,
                FPGA& fpga) {
  RunMatmul(out, in, kDim, kDim, weight_buffer, fpga);
}

void MatmulFPGA(Tensor1dFFNB& out, const Tensor1d& in,
                cl::Buffer weight_buffer, FPGA& fpga) {
  RunMatmul(out, in, kDim, kHiddenDim, weight_buffer, fpga);
}

void MatmulFPGA(Tensor1d& out, const Tensor1dFFNB& in,
                cl::Buffer weight_buffer, FPGA& fpga) {
  RunMatmul(out, in, kHiddenDim, kDim, weight_buffer, fpga);
}

void LmHeadFPGA(Tensor1dLogits& out, const Tensor1d& in, FPGA& fpga) {
  for (int i = 0; i < kDim; ++i) {
    fpga.ptr_a[i] = in[i];
  }

  fpga.q.enqueueMigrateMemObjects({fpga.buffer_a}, 0);
  for (int tile = 0; tile < (kVocabSize + kVocabTile - 1) / kVocabTile;
       ++tile) {
    fpga.kernel_matmul.setArg(1, fpga.weights.lm_head[tile]);
    fpga.kernel_matmul.setArg(3, kDim);
    fpga.kernel_matmul.setArg(4, fpga.weights.lm_head_rows[tile]);
    fpga.q.enqueueTask(fpga.kernel_matmul);
    fpga.q.enqueueMigrateMemObjects({fpga.buffer_result},
                                    CL_MIGRATE_MEM_OBJECT_HOST);
    fpga.q.finish();

    const int row_begin = tile * kVocabTile;
    for (int i = 0; i < fpga.weights.lm_head_rows[tile]; ++i) {
      out[row_begin + i] = fpga.ptr_result[i];
    }
  }
}

/* ---------------------------------  /
      Normalization Operations
/  --------------------------------- */

void RMSNormFPGA(Tensor1d& out, const Tensor1d& in, cl::Buffer weight_buffer,
                 FPGA& fpga) {
  RunRMSNorm(out, in, weight_buffer, fpga);
}

// Apply the softmax function to the input tensor.
// out[i] = exp(in[i]) / sum(exp(in[i]))
void SoftmaxFPGA(Tensor1dQKSM& out, const Tensor1dQKSM& in, int in_max_idx,
                 FPGA& fpga) {
  if (in_max_idx == -1) {
    in_max_idx = kSeqLen;
  }

  for (int i = 0; i < in_max_idx; i++) {
    fpga.ptr_a[i] = in[i];
  }
  fpga.q.enqueueMigrateMemObjects({fpga.buffer_a}, 0);
  fpga.kernel_softmax.setArg(2, in_max_idx);
  fpga.q.enqueueTask(fpga.kernel_softmax);
  fpga.q.enqueueMigrateMemObjects({fpga.buffer_result},
                                  CL_MIGRATE_MEM_OBJECT_HOST);
  fpga.q.finish();
  for (int i = 0; i < in_max_idx; i++) {
    out[i] = fpga.ptr_result[i];
  }
}

/* ---------------------------------  /
      RoPE: Position Encoding
/  --------------------------------- */

// Apply the rotary position encoding to the input tensor.
// q_out[i] = q_in[i] * cos_vec[i] - q_in[i+1] * sin_vec[i]
// q_out[i+1] = q_in[i] * sin_vec[i] + q_in[i+1] * cos_vec[i]
// k_out[i] = k_in[i] * cos_vec[i] - k_in[i+1] * sin_vec[i]
// k_out[i+1] = k_in[i] * sin_vec[i] + k_in[i+1] * cos_vec[i]
void RoPEFPGA(Tensor1d& q_out, Tensor1d& k_out, const Tensor1d& q_in,
              const Tensor1d& k_in, const Tensor1dSinCos& cos_vec,
              const Tensor1dSinCos& sin_vec, int head_begin, FPGA& fpga) {

  for (int i = 0; i < kDim; i++) {
    fpga.ptr_a[i] = q_in[i];
    fpga.ptr_b[i] = k_in[i];
  }

  for (int i = 0; i < kDim / kNHeads / 2; i++) {
    fpga.ptr_c[i] = cos_vec[i];
    fpga.ptr_d[i] = sin_vec[i];
  }

  fpga.q.enqueueMigrateMemObjects(
      {fpga.buffer_a, fpga.buffer_b, fpga.buffer_c, fpga.buffer_d}, 0);
  fpga.kernel_rope.setArg(6, head_begin);
  fpga.q.enqueueTask(fpga.kernel_rope);
  fpga.q.enqueueMigrateMemObjects({fpga.buffer_result, fpga.buffer_result2},
                                  CL_MIGRATE_MEM_OBJECT_HOST);
  fpga.q.finish();

  for (int i = head_begin; i < head_begin + kDim / kNHeads; i++) {
    q_out[i] = fpga.ptr_result[i];
    k_out[i] = fpga.ptr_result2[i];
  }
}

} // namespace llama2

#endif // USE_CPU_ONLY
