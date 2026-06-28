#ifndef DECODE_HPP_
#define DECODE_HPP_

#include "context.hpp"
#include "weight.hpp"

#ifndef USE_CPU_ONLY
#include "tensor_fpga.hpp"
#endif // USE_CPU_ONLY

namespace llama2 {

void Decode(int tok, int pos, const Tensor1d& ctx_input,
            Tensor3dCache& ctx_k_cache, Tensor3dCache& ctx_v_cache,
            Tensor1d& ctx_final_norm, const Weights& w
#ifndef USE_CPU_ONLY
            ,
            FPGA& fpga
#endif // USE_CPU_ONLY
);

} // namespace llama2

#endif // DECODE_HPP_
