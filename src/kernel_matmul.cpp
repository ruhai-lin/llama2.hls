#ifdef BUILD_DECODE_KERNEL

#include "tensor.hpp"

namespace llama2 {

constexpr int kLinearLanes = 16;
constexpr int kLinearAccBanks = 16;
constexpr int kLinearMaxIn = kFFNDim;
constexpr int kLinearMaxOut = kFFNDim;
typedef float MatmulWeightWord __attribute__((vector_size(64)));

static void kernel_matmul(float* out, int out_size, const float* in,
                          int in_size, const float* weight) {
#pragma HLS INLINE off
  // -------------------------
  // Stage A: cache input vector.
  // -------------------------
  float in_cache[kLinearMaxIn];
  float acc[kLinearAccBanks];
  
#pragma HLS ARRAY_PARTITION variable = in_cache cyclic factor = 16 dim = 1
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 1

  for (int col = 0; col < in_size; ++col) {
#pragma HLS PIPELINE II = 1
    in_cache[col] = in[col];
  }

  const MatmulWeightWord* weight_words =
      reinterpret_cast<const MatmulWeightWord*>(weight);

  // -------------------------
  // Stage B: stream rows through 16 MAC lanes.
  // -------------------------
  for (int row = 0; row < out_size; ++row) {
    for (int lane = 0; lane < kLinearLanes; ++lane) {
      #pragma HLS UNROLL
            acc[lane] = 0.0f;
          }
      
          for (int base = 0; base < in_size; base += kLinearLanes) {
      #pragma HLS PIPELINE II = 1
            const int weight_word_idx = (row * in_size + base) / kLinearLanes;
            const MatmulWeightWord weight_vec = weight_words[weight_word_idx];
      
            for (int lane = 0; lane < kLinearLanes; ++lane) {
      #pragma HLS UNROLL
              const int col = base + lane;
              acc[lane] += weight_vec[lane] * in_cache[col];
            }
          }
      
          // -------------------------
          // Stage C: tree reduction and writeback.
          // -------------------------
          float sum01 = acc[0] + acc[1];
          float sum23 = acc[2] + acc[3];
          float sum45 = acc[4] + acc[5];
          float sum67 = acc[6] + acc[7];
          float sum89 = acc[8] + acc[9];
          float sumab = acc[10] + acc[11];
          float sumcd = acc[12] + acc[13];
          float sumef = acc[14] + acc[15];
          float sum03 = sum01 + sum23;
          float sum47 = sum45 + sum67;
          float sum8b = sum89 + sumab;
          float sumcf = sumcd + sumef;
          out[row] = (sum03 + sum47) + (sum8b + sumcf);
        }
      }
      
      } // namespace llama2
      
      #endif // BUILD_DECODE_KERNEL
