#ifndef DECODE_HPP_
#define DECODE_HPP_

#include "context.hpp"
#include "weight.hpp"

#ifndef USE_CPU_ONLY
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/cl2.hpp>
#endif // USE_CPU_ONLY

namespace swan {

void Decode(int tok, int pos, const Tensor1d& ctx_input,
            Tensor3dCache& ctx_k_cache, Tensor3dCache& ctx_v_cache,
            Tensor1d& ctx_final_norm, Tensor1dLogits& ctx_logits,
            const Weights& w
#ifndef USE_CPU_ONLY
            ,
            cl::CommandQueue q, cl::Kernel kernel_decode, float* ptr_logits,
            cl::Buffer buffer_logits
#endif // USE_CPU_ONLY
);

} // namespace swan

#endif // DECODE_HPP_
