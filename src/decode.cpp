#ifdef BUILD_DECODE_KERNEL

#include "tensor.hpp"

#include <ap_int.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

namespace llama2 {
namespace {

constexpr int kHeadDim = kDim / kNumHeads;
constexpr int kAccStages = 8;
constexpr int kParameterTileRows = 64;
constexpr int kParameterTileCount = 2;
constexpr int kParameterBanks = 4;
constexpr int kBlocksPerParameterTile = kParameterTileRows / kPackedRows;
constexpr int kMaxParameterTileWords =
    kBlocksPerParameterTile * kFFNGroups * (1 + kPackedRows / 2);

using Q8ActivationBuffer = int8_t[kFFNDim];
using Q8ActivationScaleBuffer = float[kFFNGroups];
using GemvTileScores = float[kParameterTileRows];
using ParameterBeat = ap_uint<128>;
using ParameterWord = ap_uint<512>;
using CacheBeat = ap_uint<128>;
using ParameterTile =
    ParameterBeat[kParameterBanks][kMaxParameterTileWords];
using ParameterTiles = ParameterTile[kParameterTileCount];

static_assert(sizeof(ParameterBeat) == 16);
static_assert(sizeof(ParameterWord) == 64);
static_assert(sizeof(CacheBeat) == 16);
static_assert(kParameterTileRows % kPackedRows == 0);
static_assert(kParameterTileCount == 2);
static_assert(kDim % 4 == 0);
static_assert(kHeadDim % 4 == 0);

struct Q8Matrix {
  int word_offset;
  int rows;
  int group_count;
};

inline int rms_idx(int layer, int i) { return layer * kDim + i; }
inline int cache_beat_idx(int layer, int pos, int beat) {
  return (layer * kSeqLen + pos) * (kDim / 4) + beat;
}

static inline uint32_t float_to_bits(float value) {
  union {
    float value;
    uint32_t bits;
  } converter;
  converter.value = value;
  return converter.bits;
}

static inline float bits_to_float(uint32_t bits) {
  union {
    uint32_t bits;
    float value;
  } converter;
  converter.bits = bits;
  return converter.value;
}

static inline float unpack_float(const CacheBeat& beat, int lane) {
  return bits_to_float(
      beat.range((lane + 1) * 32 - 1, lane * 32).to_uint());
}

static inline void pack_float(CacheBeat& beat, int lane, float value) {
  beat.range((lane + 1) * 32 - 1, lane * 32) = float_to_bits(value);
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

static inline float unpack_scale(const ParameterWord& word, int row) {
  union {
    uint32_t bits;
    float value;
  } converter;
  converter.bits = word.range((row + 1) * 32 - 1, row * 32).to_uint();
  return converter.value;
}

static inline void read_parameter_word(const ParameterTile& tile, int word,
                                       ParameterWord& value) {
  value.range(127, 0) = tile[0][word];
  value.range(255, 128) = tile[1][word];
  value.range(383, 256) = tile[2][word];
  value.range(511, 384) = tile[3][word];
}

static void load_parameter_tile(
    const ParameterBeat* __restrict packed_params, int source_word_offset,
    int word_count, ParameterTile& __restrict tile) {
#pragma HLS INLINE off
  const int source_beat_offset = source_word_offset * kParameterBanks;
  const int beat_count = word_count * kParameterBanks;
  for (int beat = 0; beat < beat_count; ++beat) {
#pragma HLS PIPELINE II = 1
    const int bank = beat & (kParameterBanks - 1);
    const int word = beat / kParameterBanks;
    tile[bank][word] = packed_params[source_beat_offset + beat];
  }
}

template <int WeightBase>
static inline int32_t dot32(const ParameterWord& weights,
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
    const ap_int<8> weight =
        weights.range((WeightBase + lane + 1) * 8 - 1,
                      (WeightBase + lane) * 8);
    level0[lane] = weight.to_int() * static_cast<int32_t>(input[lane]);
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

static void q8_gemv_engine(const ParameterTile& tile, int block_count,
                           int group_count,
                           const Q8ActivationBuffer& activation,
                           const Q8ActivationScaleBuffer& activation_scales,
                           GemvTileScores& tile_scores) {
#pragma HLS INLINE off
  float acc[2][kPackedRows][kAccStages];
#pragma HLS ARRAY_PARTITION variable = activation cyclic factor = 32 dim = 1
#pragma HLS ARRAY_PARTITION variable = activation_scales complete dim = 1
#pragma HLS ARRAY_PARTITION variable = tile_scores cyclic factor = 16 dim = 1
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0

  const int block_word_count = group_count * (1 + kPackedRows / 2);
  for (int block = 0; block < block_count; ++block) {
    const int context = block & 1;
    const int previous_context = context ^ 1;
    const int block_word_offset = block * block_word_count;

    for (int row = 0; row < kPackedRows; ++row) {
#pragma HLS UNROLL
      clear_acc(acc[context][row]);
    }

    for (int group = 0; group < group_count; ++group) {
      ParameterWord scale_word;
      read_parameter_word(tile,
                          block_word_offset +
                              group * (1 + kPackedRows / 2),
                          scale_word);
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

      // Drain the previous context while the current block keeps accepting
      // one row-pair word per cycle.
      for (int pair = 0; pair < kPackedRows / 2; ++pair) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter false
        ParameterWord weight_word;
        read_parameter_word(tile,
                            block_word_offset +
                                group * (1 + kPackedRows / 2) + 1 + pair,
                            weight_word);
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
        acc[context][row0][stage] =
            fadd(acc[context][row0][stage], partial0);
        acc[context][row1][stage] =
            fadd(acc[context][row1][stage], partial1);

        const int score_row = group * (kPackedRows / 2) + pair;
        if (block > 0 && score_row < kPackedRows) {
          tile_scores[(block - 1) * kPackedRows + score_row] =
              reduce_acc(acc[previous_context][score_row]);
        }
      }
    }
  }

  if (block_count > 0) {
    const int final_context = (block_count - 1) & 1;
    for (int row = 0; row < kPackedRows; ++row) {
#pragma HLS PIPELINE II = 1
      tile_scores[(block_count - 1) * kPackedRows + row] =
          reduce_acc(acc[final_context][row]);
    }
  }
}

static void compute_parameter_tile(
    const ParameterTile& tile, int block_count, int group_count,
    const Q8ActivationBuffer& activation,
    const Q8ActivationScaleBuffer& activation_scales,
    GemvTileScores& tile_scores) {
#pragma HLS INLINE off
  q8_gemv_engine(tile, block_count, group_count, activation, activation_scales,
                 tile_scores);
}

static void q8_gemv_step(
    const ParameterBeat* packed_params, int load_word_offset,
    int load_word_count, ParameterTile& load_tile,
    const ParameterTile& compute_tile, int compute_block_count,
    int compute_group_count,
    const Q8ActivationBuffer& activation,
    const Q8ActivationScaleBuffer& activation_scales,
    GemvTileScores& tile_scores) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
  load_parameter_tile(packed_params, load_word_offset, load_word_count,
                      load_tile);
  compute_parameter_tile(compute_tile, compute_block_count, compute_group_count,
                         activation, activation_scales, tile_scores);
}

static void q8_gemv_tile(
    ParameterTiles& tiles, const ParameterBeat* packed_params,
    int load_word_offset, int load_word_count, int load_tile,
    int compute_block_count, int compute_group_count,
    const Q8ActivationBuffer& activation,
    const Q8ActivationScaleBuffer& activation_scales,
    GemvTileScores& tile_scores) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = tile_scores cyclic factor = 16 dim = 1
#pragma HLS ALLOCATION function instances = q8_gemv_step limit = 1

  if (load_tile == 0) {
    q8_gemv_step(packed_params, load_word_offset, load_word_count, tiles[0],
                 tiles[1], compute_block_count, compute_group_count,
                 activation, activation_scales, tile_scores);
  } else {
    q8_gemv_step(packed_params, load_word_offset, load_word_count, tiles[1],
                 tiles[0], compute_block_count, compute_group_count,
                 activation, activation_scales, tile_scores);
  }
}

static inline int first_tile_word_count(const Q8Matrix& matrix) {
  const int block_count = matrix.rows / kPackedRows;
  const int tile_block_count =
      block_count < kBlocksPerParameterTile ? block_count
                                            : kBlocksPerParameterTile;
  return tile_block_count * matrix.group_count *
         (1 + kPackedRows / 2);
}

static uint32_t q8_linear(
    ParameterTiles& parameter_tiles, float* output, const Q8Matrix& matrix,
    const Q8ActivationBuffer& activation,
    const Q8ActivationScaleBuffer& activation_scales,
    const ParameterBeat* packed_params, bool has_prefetched_tile,
    int prefetched_tile,
    const Q8Matrix& next_matrix, int& next_prefetched_tile,
    bool argmax_mode = false) {
#pragma HLS INLINE
  const int block_word_count =
      matrix.group_count * (1 + kPackedRows / 2);
  const int block_count = matrix.rows / kPackedRows;
  float best_score = -FLT_MAX;
  uint32_t best_token = 0;
  GemvTileScores tile_scores;
#pragma HLS ARRAY_PARTITION variable = tile_scores cyclic factor = 16 dim = 1

  int current_tile = has_prefetched_tile ? prefetched_tile : 0;
  if (!has_prefetched_tile) {
    volatile int runtime_load_word_offset = matrix.word_offset;
    volatile int runtime_load_word_count = first_tile_word_count(matrix);
    volatile int runtime_load_tile = 0;
    volatile int runtime_compute_block_count = 0;
    volatile int runtime_compute_group_count = matrix.group_count;
    q8_gemv_tile(
        parameter_tiles, packed_params, runtime_load_word_offset,
        runtime_load_word_count, runtime_load_tile, runtime_compute_block_count,
        runtime_compute_group_count, activation, activation_scales,
        tile_scores);
  }

  for (int tile_block = 0; tile_block < block_count;
       tile_block += kBlocksPerParameterTile) {
    const int remaining_blocks = block_count - tile_block;
    const int current_block_count =
        remaining_blocks < kBlocksPerParameterTile
            ? remaining_blocks
            : kBlocksPerParameterTile;
    const int next_tile_block = tile_block + kBlocksPerParameterTile;
    const int next_remaining_blocks = block_count - next_tile_block;
    const int next_block_count =
        next_remaining_blocks <= 0
            ? 0
            : (next_remaining_blocks < kBlocksPerParameterTile
                   ? next_remaining_blocks
                   : kBlocksPerParameterTile);
    const int load_tile = current_tile ^ 1;
    const bool load_next_matrix = next_block_count == 0;
    const int load_word_offset =
        load_next_matrix
            ? next_matrix.word_offset
            : matrix.word_offset + next_tile_block * block_word_count;
    const int load_word_count =
        load_next_matrix
            ? first_tile_word_count(next_matrix)
            : next_block_count * block_word_count;

    // Runtime-valued arguments keep every matrix shape on the same physical
    // ping/pong loader and GEMV engine.
    volatile int runtime_load_word_offset = load_word_offset;
    volatile int runtime_load_word_count = load_word_count;
    volatile int runtime_load_tile = load_tile;
    volatile int runtime_compute_block_count = current_block_count;
    volatile int runtime_compute_group_count = matrix.group_count;
    q8_gemv_tile(
        parameter_tiles, packed_params, runtime_load_word_offset,
        runtime_load_word_count, runtime_load_tile, runtime_compute_block_count,
        runtime_compute_group_count, activation, activation_scales,
        tile_scores);

    for (int block = 0; block < current_block_count; ++block) {
      for (int row = 0; row < kPackedRows; ++row) {
#pragma HLS PIPELINE II = 1
        const int tile_row = block * kPackedRows + row;
        const int output_row = tile_block * kPackedRows + tile_row;
        const float score = tile_scores[tile_row];
        if (argmax_mode) {
          if (score > best_score) {
            best_score = score;
            best_token = static_cast<uint32_t>(output_row);
          }
        } else {
          output[output_row] = score;
        }
      }
    }
    current_tile = load_tile;
  }
  next_prefetched_tile = current_tile;
  return best_token;
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
                 const float* sin_table, int pos, bool load_position) {
#pragma HLS INLINE off
  static Tensor1dSinCos cos_pos;
  static Tensor1dSinCos sin_pos;
  if (load_position) {
    for (int i = 0; i < kSinCosTable; ++i) {
#pragma HLS PIPELINE II = 1
      cos_pos[i] = cos_table[pos * kSinCosTable + i];
      sin_pos[i] = sin_table[pos * kSinCosTable + i];
    }
  }

  for (int head = 0; head < kNumHeads; ++head) {
    const int head_begin = head * kHeadDim;
    for (int i = 0; i < kHeadDim / 2; ++i) {
#pragma HLS PIPELINE II = 1
      const int i0 = head_begin + i * 2;
      const int i1 = i0 + 1;
      const float c = cos_pos[i];
      const float s = sin_pos[i];
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

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

static void silu_mul(Tensor1dFFNB& out, const Tensor1dFFNB& gate,
                     const Tensor1dFFNB& up) {
  for (int i = 0; i < kFFNDim; ++i) {
#pragma HLS PIPELINE II = 1
    out[i] = silu(gate[i]) * up[i];
  }
}

static void store_cache_row(CacheBeat* cache, int layer, int pos,
                            const Tensor1d& row) {
#pragma HLS INLINE off
  for (int beat = 0; beat < kDim / 4; ++beat) {
#pragma HLS PIPELINE II = 1
    CacheBeat packed = 0;
    for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
      pack_float(packed, lane, row[beat * 4 + lane]);
    }
    cache[cache_beat_idx(layer, pos, beat)] = packed;
  }
}

static void flash_attention(Tensor1d& output, const Tensor1d& query, int layer,
                            int pos, const CacheBeat* k_cache,
                            const CacheBeat* v_cache) {
#pragma HLS INLINE off
  float running_max[kNumHeads];
  float running_sum[kNumHeads];
  float alpha[kNumHeads];
  float beta[kNumHeads];
  float weighted_value[kDim];
  float dot_acc[kNumHeads][kAccStages];
#pragma HLS ARRAY_PARTITION variable = running_max complete dim = 1
#pragma HLS ARRAY_PARTITION variable = running_sum complete dim = 1
#pragma HLS ARRAY_PARTITION variable = alpha complete dim = 1
#pragma HLS ARRAY_PARTITION variable = beta complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weighted_value cyclic factor = 4 dim = 1
#pragma HLS ARRAY_PARTITION variable = dot_acc complete dim = 0

  for (int head = 0; head < kNumHeads; ++head) {
#pragma HLS UNROLL
    running_max[head] = -FLT_MAX;
    running_sum[head] = 0.0f;
    alpha[head] = 0.0f;
    beta[head] = 0.0f;
  }
  for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
    weighted_value[i] = 0.0f;
  }

  const float score_scale = 1.0f / sqrtf(static_cast<float>(kHeadDim));
  for (int t = 0; t <= pos; ++t) {
    for (int head = 0; head < kNumHeads; ++head) {
#pragma HLS UNROLL
      clear_acc(dot_acc[head]);
    }

    for (int beat = 0; beat < kDim / 4; ++beat) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = dot_acc inter false
      const int head = beat / (kHeadDim / 4);
      const int head_beat = beat % (kHeadDim / 4);
      const int stage = head_beat & (kAccStages - 1);
      const CacheBeat packed =
          k_cache[cache_beat_idx(layer, t, beat)];
      float product[4];
#pragma HLS ARRAY_PARTITION variable = product complete dim = 1
      for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
        const int i = beat * 4 + lane;
        product[lane] = query[i] * unpack_float(packed, lane);
      }
      const float partial =
          fadd(fadd(product[0], product[1]), fadd(product[2], product[3]));
      dot_acc[head][stage] = fadd(dot_acc[head][stage], partial);
    }

    for (int head = 0; head < kNumHeads; ++head) {
#pragma HLS PIPELINE II = 1
      const float score = reduce_acc(dot_acc[head]) * score_scale;
      if (t == 0) {
        running_max[head] = score;
        running_sum[head] = 1.0f;
        alpha[head] = 0.0f;
        beta[head] = 1.0f;
      } else if (score > running_max[head]) {
        const float rescale = expf(running_max[head] - score);
        running_max[head] = score;
        running_sum[head] = running_sum[head] * rescale + 1.0f;
        alpha[head] = rescale;
        beta[head] = 1.0f;
      } else {
        const float weight = expf(score - running_max[head]);
        running_sum[head] += weight;
        alpha[head] = 1.0f;
        beta[head] = weight;
      }
    }

    for (int beat = 0; beat < kDim / 4; ++beat) {
#pragma HLS PIPELINE II = 1
      const int head = beat / (kHeadDim / 4);
      const CacheBeat packed =
          v_cache[cache_beat_idx(layer, t, beat)];
      for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
        const int i = beat * 4 + lane;
        weighted_value[i] =
            weighted_value[i] * alpha[head] +
            unpack_float(packed, lane) * beta[head];
      }
    }
  }

  for (int beat = 0; beat < kDim / 4; ++beat) {
#pragma HLS PIPELINE II = 1
    const int head = beat / (kHeadDim / 4);
    for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
      const int i = beat * 4 + lane;
      output[i] = weighted_value[i] / running_sum[head];
    }
  }
}

static void attn(Tensor1d& hidden, int layer, int pos,
                 const float* rms_att_w,
                 const ParameterBeat* packed_params,
                 const float* cos_table, const float* sin_table,
                 CacheBeat* k_cache, CacheBeat* v_cache,
                 ParameterTiles& parameter_tiles,
                 int& prefetched_tile) {
#pragma HLS INLINE

  Tensor1d attn_input;
  Tensor1d attn_norm;
  Tensor1d q;
  Tensor1d k;
  Tensor1d v;
  Tensor1d q_rot;
  Tensor1d k_rot;
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

  const Q8Matrix wq_matrix = {
      kPackedAttnWqWordOffset + layer * kPackedAttnLayerWords, kDim,
      kDimGroups};
  const Q8Matrix wk_matrix = {
      kPackedAttnWkWordOffset + layer * kPackedAttnLayerWords, kDim,
      kDimGroups};
  const Q8Matrix wv_matrix = {
      kPackedAttnWvWordOffset + layer * kPackedAttnLayerWords, kDim,
      kDimGroups};
  const Q8Matrix wo_matrix = {
      kPackedAttnWoWordOffset + layer * kPackedAttnLayerWords, kDim,
      kDimGroups};
  const Q8Matrix w1_matrix = {
      kPackedFFNW1WordOffset + layer * kPackedFFNALayerWords, kFFNDim,
      kDimGroups};

  for (int i = 0; i < kDim; ++i) {
#pragma HLS PIPELINE II = 1
    attn_input[i] = hidden[i];
  }

  rmsnorm(attn_norm, attn_input, &rms_att_w[rms_idx(layer, 0)]);
  quantize(attn_norm_q8, attn_norm_scales, attn_norm);

  int next_prefetched_tile = 0;
  q8_linear(parameter_tiles, q, wq_matrix, attn_norm_q8, attn_norm_scales,
            packed_params, layer != 0, prefetched_tile, wk_matrix,
            next_prefetched_tile);
  prefetched_tile = next_prefetched_tile;

  q8_linear(parameter_tiles, k, wk_matrix, attn_norm_q8, attn_norm_scales,
            packed_params, true, prefetched_tile, wv_matrix,
            next_prefetched_tile);
  prefetched_tile = next_prefetched_tile;

  q8_linear(parameter_tiles, v, wv_matrix, attn_norm_q8, attn_norm_scales,
            packed_params, true, prefetched_tile, wo_matrix,
            next_prefetched_tile);
  prefetched_tile = next_prefetched_tile;

  rope(q_rot, k_rot, q, k, cos_table, sin_table, pos, layer == 0);

  store_cache_row(k_cache, layer, pos, k_rot);
  store_cache_row(v_cache, layer, pos, v);
  flash_attention(attn_val, q_rot, layer, pos, k_cache, v_cache);

  quantize(attn_value_q8, attn_value_scales, attn_val);
  q8_linear(parameter_tiles, attn_out, wo_matrix, attn_value_q8,
            attn_value_scales, packed_params, true, prefetched_tile, w1_matrix,
            next_prefetched_tile);
  prefetched_tile = next_prefetched_tile;

  add(hidden, attn_input, attn_out);
}

static void ffn(Tensor1d& hidden, int layer, const float* rms_ffn_w,
                const ParameterBeat* packed_params,
                ParameterTiles& parameter_tiles, int& prefetched_tile) {
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

  const Q8Matrix w1_matrix = {
      kPackedFFNW1WordOffset + layer * kPackedFFNALayerWords, kFFNDim,
      kDimGroups};
  const Q8Matrix w3_matrix = {
      kPackedFFNW3WordOffset + layer * kPackedFFNALayerWords, kFFNDim,
      kDimGroups};
  const Q8Matrix w2_matrix = {
      kPackedFFNW2WordOffset + layer * kPackedFFNBLayerWords, kDim,
      kFFNGroups};
  const Q8Matrix next_matrix =
      layer + 1 < kNumLayers
          ? Q8Matrix{kPackedAttnWqWordOffset +
                         (layer + 1) * kPackedAttnLayerWords,
                     kDim, kDimGroups}
          : Q8Matrix{kPackedTokWordOffset, kVocabSize, kDimGroups};

  rmsnorm(ffn_norm, hidden, &rms_ffn_w[rms_idx(layer, 0)]);
  quantize(ffn_norm_q8, ffn_norm_scales, ffn_norm);

  int next_prefetched_tile = 0;
  q8_linear(parameter_tiles, w1, w1_matrix, ffn_norm_q8, ffn_norm_scales,
            packed_params, true, prefetched_tile, w3_matrix,
            next_prefetched_tile);
  prefetched_tile = next_prefetched_tile;

  q8_linear(parameter_tiles, w3, w3_matrix, ffn_norm_q8, ffn_norm_scales,
            packed_params, true, prefetched_tile, w2_matrix,
            next_prefetched_tile);
  prefetched_tile = next_prefetched_tile;

  silu_mul(dot, w1, w3);
  quantize(ffn_product_q8, ffn_product_scales, dot);

  q8_linear(parameter_tiles, out, w2_matrix, ffn_product_q8,
            ffn_product_scales, packed_params, true, prefetched_tile,
            next_matrix, next_prefetched_tile);
  prefetched_tile = next_prefetched_tile;

  add_inplace(hidden, out);
}

static uint32_t lm_head(const Tensor1d& hidden,
                        const ParameterBeat* packed_params,
                        const float* rms_final,
                        ParameterTiles& parameter_tiles, int prefetched_tile) {
#pragma HLS INLINE

  Tensor1d final_norm;
  Q8ActivationBuffer final_norm_q8;
  Q8ActivationScaleBuffer final_norm_scales;
#pragma HLS ARRAY_PARTITION variable = final_norm_q8 cyclic factor = 32 dim = 1
#pragma HLS ARRAY_PARTITION variable = final_norm_scales complete dim = 1

  rmsnorm(final_norm, hidden, rms_final);
  quantize(final_norm_q8, final_norm_scales, final_norm);

  const Q8Matrix lm_matrix = {kPackedTokWordOffset, kVocabSize, kDimGroups};
  const Q8Matrix no_next_matrix = {0, 0, 0};
  int unused_prefetched_tile = 0;
  return q8_linear(parameter_tiles, final_norm, lm_matrix, final_norm_q8,
                   final_norm_scales, packed_params, true, prefetched_tile,
                   no_next_matrix, unused_prefetched_tile, true);
}

} // namespace
} // namespace llama2

using namespace llama2;

extern "C" {

void decode(int token, int pos, const float* tok_emb_table,
            const ParameterBeat* packed_params,
            const float* rms_att_w, const float* rms_ffn_w,
            const float* rms_final, const float* cos_table,
            const float* sin_table, CacheBeat* k_cache, CacheBeat* v_cache,
            uint32_t* next_token) {
// -------------------------
// Stage 0: AXI interfaces.
// -------------------------
#pragma HLS INTERFACE m_axi port = tok_emb_table bundle = gmem
#pragma HLS INTERFACE m_axi port = packed_params bundle = gmem \
    max_read_burst_length = 256 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = rms_att_w bundle = gmem
#pragma HLS INTERFACE m_axi port = rms_ffn_w bundle = gmem
#pragma HLS INTERFACE m_axi port = rms_final bundle = gmem
#pragma HLS INTERFACE m_axi port = cos_table bundle = gmem
#pragma HLS INTERFACE m_axi port = sin_table bundle = gmem
#pragma HLS INTERFACE m_axi port = k_cache bundle = gmem
#pragma HLS INTERFACE m_axi port = v_cache bundle = gmem
#pragma HLS INTERFACE m_axi port = next_token bundle = gmem
#pragma HLS ALLOCATION function instances = q8_gemv_tile limit = 1
#pragma HLS ALLOCATION function instances = q8_gemv_step limit = 1
#pragma HLS ALLOCATION function instances = load_parameter_tile limit = 1
#pragma HLS ALLOCATION function instances = compute_parameter_tile limit = 1
#pragma HLS ALLOCATION function instances = q8_gemv_engine limit = 1

  Tensor1d hidden;
  ParameterTiles parameter_tiles;
#pragma HLS ARRAY_PARTITION variable = parameter_tiles complete dim = 1
#pragma HLS ARRAY_PARTITION variable = parameter_tiles complete dim = 2
#pragma HLS BIND_STORAGE variable = parameter_tiles type = ram_2p impl = uram
  load_token(hidden, token, tok_emb_table);

  int prefetched_tile = 0;
  for (int layer = 0; layer < kNumLayers; ++layer) {
    attn(hidden, layer, pos, rms_att_w, packed_params, cos_table, sin_table,
         k_cache, v_cache, parameter_tiles, prefetched_tile);
    ffn(hidden, layer, rms_ffn_w, packed_params, parameter_tiles,
        prefetched_tile);
  }

  *next_token =
      lm_head(hidden, packed_params, rms_final, parameter_tiles, prefetched_tile);
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
