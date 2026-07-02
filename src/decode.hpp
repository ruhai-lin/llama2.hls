#ifndef DECODE_HPP_
#define DECODE_HPP_

#include "context.hpp"
#include "weight.hpp"

#include <stdint.h>

#ifndef USE_CPU_ONLY
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/cl2.hpp>
#endif // USE_CPU_ONLY

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
);

} // namespace llama2

#endif // DECODE_HPP_
