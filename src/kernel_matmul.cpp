#ifndef USE_CPU_ONLY

#include <ap_int.h>
#include <stdint.h>

#define MAX_DATA_SIZE 1024
#define LANES 16

static float bits_to_float(ap_uint<32> bits) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = bits.to_uint();
  return v.f;
}

static float reduce16(float x[LANES]) {
#pragma HLS INLINE
  float s0[8];
  float s1[4];
  float s2[2];
#pragma HLS ARRAY_PARTITION variable = s0 complete
#pragma HLS ARRAY_PARTITION variable = s1 complete
#pragma HLS ARRAY_PARTITION variable = s2 complete
#pragma HLS BIND_OP variable = s0 op = fadd impl = fulldsp latency = 6
#pragma HLS BIND_OP variable = s1 op = fadd impl = fulldsp latency = 6
#pragma HLS BIND_OP variable = s2 op = fadd impl = fulldsp latency = 6

reduce_0:
  for (int i = 0; i < 8; ++i) {
#pragma HLS UNROLL
    s0[i] = x[2 * i] + x[2 * i + 1];
  }
reduce_1:
  for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
    s1[i] = s0[2 * i] + s0[2 * i + 1];
  }
reduce_2:
  for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
    s2[i] = s1[2 * i] + s1[2 * i + 1];
  }
  return s2[0] + s2[1];
}

static float add_sum(float sum, float value) {
#pragma HLS INLINE
  float next = sum + value;
#pragma HLS BIND_OP variable = next op = fadd impl = fulldsp latency = 6
  return next;
}

static void cache_input(float* i_vec, float vec_local[MAX_DATA_SIZE],
                        int vec_size) {
cache_input_loop:
  for (int i = 0; i < vec_size; ++i) {
#pragma HLS PIPELINE II = 1
    vec_local[i] = i_vec[i];
  }
}

static float dot_row(ap_uint<512>* i_mat, float vec_local[MAX_DATA_SIZE],
                     int row, int vec_size) {
  float sum = 0.0f;
  const int tiles = vec_size / LANES;
  const int row_base = row * tiles;

tile_loop:
  for (int tile = 0; tile < tiles; ++tile) {
#pragma HLS PIPELINE II = 1
    ap_uint<512> packed = i_mat[row_base + tile];
    float products[LANES];
#pragma HLS ARRAY_PARTITION variable = products complete

lane_loop:
    for (int lane = 0; lane < LANES; ++lane) {
#pragma HLS UNROLL
      ap_uint<32> weight_bits = packed.range(32 * lane + 31, 32 * lane);
      float weight = bits_to_float(weight_bits);
      products[lane] = vec_local[tile * LANES + lane] * weight;
    }
    sum = add_sum(sum, reduce16(products));
  }

  return sum;
}

static void compute_rows(ap_uint<512>* i_mat, float* o_vec,
                         float vec_local[MAX_DATA_SIZE], int vec_size,
                         int col_size) {
row_loop:
  for (int row = 0; row < col_size; ++row) {
    o_vec[row] = dot_row(i_mat, vec_local, row, vec_size);
  }
}

extern "C" {
void kernel_matmul(float* i_vec, ap_uint<512>* i_mat, float* o_vec,
                   int vec_size, int col_size) {
#pragma HLS INTERFACE m_axi port = i_vec bundle = gmem0
#pragma HLS INTERFACE m_axi port = i_mat bundle = gmem1
#pragma HLS INTERFACE m_axi port = o_vec bundle = gmem0

  float vec_local[MAX_DATA_SIZE];
#pragma HLS ARRAY_PARTITION variable = vec_local cyclic factor = LANES

  // Stage A: cache input
  cache_input(i_vec, vec_local, vec_size);

  // Stage B: compute rows
  compute_rows(i_mat, o_vec, vec_local, vec_size, col_size);
}
}

#endif // USE_CPU_ONLY
