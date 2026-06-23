#include <cstring>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "context.hpp"
#include "decode.hpp"
#include "vocab.hpp"
#include "weight.hpp"

#ifndef USE_CPU_ONLY
#include <stdlib.h>

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/cl2.hpp>

#define OCL_CHECK(error, call)                                             \
  call;                                                                    \
  if (error != CL_SUCCESS) {                                               \
    printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, \
           __LINE__, error);                                               \
    exit(EXIT_FAILURE);                                                    \
  }

template <typename T>
struct aligned_allocator {
  using value_type = T;
  T* allocate(std::size_t num) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, num * sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t num) { free(p); }
};

template <typename Scalar, typename T>
std::vector<Scalar, aligned_allocator<Scalar>> FlattenWeights(const T& data,
                                                              std::size_t count) {
  std::vector<Scalar, aligned_allocator<Scalar>> flat(count);
  std::memcpy(flat.data(), reinterpret_cast<const Scalar*>(&data),
              count * sizeof(Scalar));
  return flat;
}

template <typename T>
cl::Buffer CreateKernelArgBuffer(const cl::Context& context,
                                 const cl::Kernel& kernel, unsigned int argidx,
                                 cl_mem_flags flags, std::size_t bytes,
                                 T* host_ptr, cl_int* err) {
  (void)kernel;
  (void)argidx;
  return cl::Buffer(context, flags | CL_MEM_USE_HOST_PTR, bytes, host_ptr, err);
}
#endif // USE_CPU_ONLY

// Command line arguments.
struct Args {
  std::string weight_path = "./model/stories15M_q8.bin";
  std::string vocab_path = "./model/tokenizer.bin";
  uint64_t max_seq = 256;
  float temp = 0.5;
  bool color = false;
  bool print_softmax = false;
  bool log = false;
  bool help = false;
};

// Parse the command line arguments.
void ParseArgument(int argc, char* argv[], Args& args) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--weight_path") == 0 && i + 1 < argc) {
      args.weight_path = argv[++i];
    } else if (std::strcmp(argv[i], "--vocab_path") == 0 && i + 1 < argc) {
      args.vocab_path = argv[++i];
    } else if (std::strcmp(argv[i], "--max_seq") == 0 && i + 1 < argc) {
      args.max_seq = std::stoull(argv[++i]);
    } else if (std::strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
      args.temp = std::stof(argv[++i]);
    } else if (std::strcmp(argv[i], "--color") == 0) {
      args.color = true;
    } else if (std::strcmp(argv[i], "--print_softmax") == 0) {
      args.print_softmax = true;
    } else if (std::strcmp(argv[i], "--log") == 0) {
      args.log = true;
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      args.help = true;
    } else {
      std::cerr << "[ERROR] Unknown Option: " << argv[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

// Random Sampling
int SelectFromLogits(const llama2::Tensor1dLogits& prob_dist) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);

  const int vocab_size = llama2::kVocabSize;
  float rand = dis(gen);

  float cdf = 0.0;
  for (size_t i = 0; i < vocab_size; ++i) {
    cdf += prob_dist[i];
    if (rand < cdf) {
      return i;
    }
  }

  // in case of rounding errors
  return vocab_size - 1;
}

int main(int argc, char* argv[]) {
  // 1. Parse arguments.
  Args args;
  ParseArgument(argc, argv, args);

  if (args.help) {
    std::cout << "Usage: " << argv[0] << " [options]" << std::endl
              << "Options:" << std::endl
              << "  --weight_path   : Weight file path" << std::endl
              << "  --vocab_path    : Tokenizer file path" << std::endl
              << "  --max_seq       : Maximum sequence length" << std::endl
              << "  --temp          : Temperature for sampling" << std::endl
              << "  --color         : Enable color output" << std::endl
              << "  --log           : Enable log output" << std::endl
              << "  --help, -h      : Show this help message" << std::endl;
    return 0;
  }

  // 2. Print hyper parameters.
  std::cout << "Hyper Parameters" << std::endl
            << "  dim       : " << llama2::kDim << std::endl
            << "  ffn_dim   : " << llama2::kFFNDim << std::endl
            << "  n_layers  : " << llama2::kNumLayers << std::endl
            << "  n_heads   : " << llama2::kNumHeads << std::endl
            << "  n_kv_heads: " << llama2::kNumKVHeads << std::endl
            << "  vocab_size: " << llama2::kVocabSize << std::endl
            << "  seq_len   : " << llama2::kSeqLen << std::endl;

  // 3. Load model parameters.
  std::ifstream weight_fs(args.weight_path, std::ios::in | std::ios::binary);
  if (!weight_fs) {
    std::cout << "Failed to open: " << args.weight_path << std::endl;
    return EXIT_FAILURE;
  }
  static llama2::Weights weights;
  static llama2::Tensor2dTok tok_emb_table; // [vocab_size, dim]
  llama2::LoadWeights(weights, tok_emb_table, weight_fs);
  weight_fs.close();

  // 4. Load vocabrary.
  std::ifstream vocab_fs(args.vocab_path, std::ios::in | std::ios::binary);
  if (!vocab_fs) {
    std::cout << "Failed to open: " << args.vocab_path << std::endl;
    return EXIT_FAILURE;
  }
  static llama2::Vocab vocab;
  const int vocab_size = llama2::kVocabSize;
  llama2::ResizeVocab(vocab, vocab_size);
  llama2::LoadVocab(vocab, vocab_fs);
  vocab_fs.close();

#ifndef USE_CPU_ONLY
  // 5. OpenCL Settings
  std::string xclbinFilename = "./binary_container_1.bin";

  constexpr std::size_t tok_emb_count = llama2::kVocabSize * llama2::kDim;
  constexpr std::size_t tok_emb_s_count =
      llama2::kVocabSize * llama2::kDimGroups;
  constexpr std::size_t rms_count = llama2::kNumLayers * llama2::kDim;
  constexpr std::size_t attn_count =
      llama2::kNumLayers * llama2::kDim * llama2::kDim;
  constexpr std::size_t attn_s_count =
      llama2::kNumLayers * llama2::kDim * llama2::kDimGroups;
  constexpr std::size_t ffn_a_count =
      llama2::kNumLayers * llama2::kFFNDim * llama2::kDim;
  constexpr std::size_t ffn_a_s_count =
      llama2::kNumLayers * llama2::kFFNDim * llama2::kDimGroups;
  constexpr std::size_t ffn_b_count =
      llama2::kNumLayers * llama2::kDim * llama2::kFFNDim;
  constexpr std::size_t ffn_b_s_count =
      llama2::kNumLayers * llama2::kDim * llama2::kFFNGroups;
  constexpr std::size_t rms_final_count = llama2::kDim;
  constexpr std::size_t sincos_count = llama2::kSeqLen * llama2::kSinCosTable;
  constexpr std::size_t cache_count =
      llama2::kNumLayers * llama2::kSeqLen * llama2::kDim;

  auto tok_emb_host = FlattenWeights<float>(tok_emb_table, tok_emb_count);
  auto tok_emb_q_host = FlattenWeights<int8_t>(weights.tok_emb_q, tok_emb_count);
  auto tok_emb_s_host = FlattenWeights<float>(weights.tok_emb_s, tok_emb_s_count);
  auto rms_att_host = FlattenWeights<float>(weights.rms_att_w, rms_count);
  auto attn_wq_host = FlattenWeights<int8_t>(weights.attn_wq, attn_count);
  auto attn_wq_s_host = FlattenWeights<float>(weights.attn_wq_s, attn_s_count);
  auto attn_wk_host = FlattenWeights<int8_t>(weights.attn_wk, attn_count);
  auto attn_wk_s_host = FlattenWeights<float>(weights.attn_wk_s, attn_s_count);
  auto attn_wv_host = FlattenWeights<int8_t>(weights.attn_wv, attn_count);
  auto attn_wv_s_host = FlattenWeights<float>(weights.attn_wv_s, attn_s_count);
  auto attn_wo_host = FlattenWeights<int8_t>(weights.attn_wo, attn_count);
  auto attn_wo_s_host = FlattenWeights<float>(weights.attn_wo_s, attn_s_count);
  auto rms_ffn_host = FlattenWeights<float>(weights.rms_ffn_w, rms_count);
  auto ffn_w1_host = FlattenWeights<int8_t>(weights.ffn_w1, ffn_a_count);
  auto ffn_w1_s_host = FlattenWeights<float>(weights.ffn_w1_s, ffn_a_s_count);
  auto ffn_w2_host = FlattenWeights<int8_t>(weights.ffn_w2, ffn_b_count);
  auto ffn_w2_s_host = FlattenWeights<float>(weights.ffn_w2_s, ffn_b_s_count);
  auto ffn_w3_host = FlattenWeights<int8_t>(weights.ffn_w3, ffn_a_count);
  auto ffn_w3_s_host = FlattenWeights<float>(weights.ffn_w3_s, ffn_a_s_count);
  auto rms_final_host = FlattenWeights<float>(weights.rms_final, rms_final_count);
  auto cos_host = FlattenWeights<float>(weights.cos_table, sincos_count);
  auto sin_host = FlattenWeights<float>(weights.sin_table, sincos_count);
  std::vector<float, aligned_allocator<float>> k_cache_host(cache_count, 0.0f);
  std::vector<float, aligned_allocator<float>> v_cache_host(cache_count, 0.0f);
  std::vector<uint32_t, aligned_allocator<uint32_t>> next_host(1, 0);

  std::vector<cl::Device> devices;
  cl_int err;
  cl::Context context;
  cl::CommandQueue q;
  cl::Kernel kernel_decode;
  cl::Program program;
  std::vector<cl::Platform> platforms;
  bool found_device = false;

  // traversing all Platforms To find Xilinx Platform and targeted
  // Device in Xilinx Platform
  cl::Platform::get(&platforms);
  for (size_t i = 0; (i < platforms.size()) & (found_device == false); i++) {
    cl::Platform platform = platforms[i];
    std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
    if (platformName == "Xilinx") {
      devices.clear();
      platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
      if (devices.size()) {
        found_device = true;
        break;
      }
    }
  }
  if (found_device == false) {
    std::cout << "Error: Unable to find Target Device " << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "INFO: Reading " << xclbinFilename << std::endl;
  FILE* fp;
  if ((fp = fopen(xclbinFilename.c_str(), "r")) == nullptr) {
    printf("ERROR: %s xclbin not available please build\n",
           xclbinFilename.c_str());
    exit(EXIT_FAILURE);
  }
  // Load xclbin
  std::cout << "Loading: '" << xclbinFilename << "'\n";
  std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
  bin_file.seekg(0, bin_file.end);
  unsigned nb = bin_file.tellg();
  bin_file.seekg(0, bin_file.beg);
  char* buf = new char[nb];
  bin_file.read(buf, nb);

  // Creating Program from Binary File
  cl::Program::Binaries bins;
  bins.push_back({buf, nb});
  bool valid_device = false;
  for (unsigned int i = 0; i < devices.size(); i++) {
    auto device = devices[i];
    // Creating Context and Command Queue for selected Device
    OCL_CHECK(err,
              context = cl::Context(device, nullptr, nullptr, nullptr, &err));
    OCL_CHECK(err, q = cl::CommandQueue(context, device,
                                        CL_QUEUE_PROFILING_ENABLE, &err));
    std::cout << "Trying to program device[" << i
              << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    program = cl::Program(context, {device}, bins, nullptr, &err);
    if (err != CL_SUCCESS) {
      std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
    } else {
      std::cout << "Device[" << i << "]: program successful!\n";
      OCL_CHECK(err,
                kernel_decode = cl::Kernel(program, "kernel_decode", &err));
      std::cout << "load : kernel_decode" << std::endl;
      valid_device = true;
      break; // we break because we found a valid device
    }
  }
  if (!valid_device) {
    std::cout << "Failed to program any device found, exit!\n";
    exit(EXIT_FAILURE);
  }

  OCL_CHECK(err, cl::Buffer buffer_tok_emb = CreateKernelArgBuffer(
                     context, kernel_decode, 2, CL_MEM_READ_ONLY,
                     tok_emb_count * sizeof(float), tok_emb_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_tok_emb_q = CreateKernelArgBuffer(
                     context, kernel_decode, 3, CL_MEM_READ_ONLY,
                     tok_emb_count * sizeof(int8_t), tok_emb_q_host.data(),
                     &err));
  OCL_CHECK(err, cl::Buffer buffer_tok_emb_s = CreateKernelArgBuffer(
                     context, kernel_decode, 4, CL_MEM_READ_ONLY,
                     tok_emb_s_count * sizeof(float), tok_emb_s_host.data(),
                     &err));
  OCL_CHECK(err, cl::Buffer buffer_rms_att = CreateKernelArgBuffer(
                     context, kernel_decode, 5, CL_MEM_READ_ONLY,
                     rms_count * sizeof(float), rms_att_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_attn_wq = CreateKernelArgBuffer(
                     context, kernel_decode, 6, CL_MEM_READ_ONLY,
                     attn_count * sizeof(int8_t), attn_wq_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_attn_wq_s = CreateKernelArgBuffer(
                     context, kernel_decode, 7, CL_MEM_READ_ONLY,
                     attn_s_count * sizeof(float), attn_wq_s_host.data(),
                     &err));
  OCL_CHECK(err, cl::Buffer buffer_attn_wk = CreateKernelArgBuffer(
                     context, kernel_decode, 8, CL_MEM_READ_ONLY,
                     attn_count * sizeof(int8_t), attn_wk_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_attn_wk_s = CreateKernelArgBuffer(
                     context, kernel_decode, 9, CL_MEM_READ_ONLY,
                     attn_s_count * sizeof(float), attn_wk_s_host.data(),
                     &err));
  OCL_CHECK(err, cl::Buffer buffer_attn_wv = CreateKernelArgBuffer(
                     context, kernel_decode, 10, CL_MEM_READ_ONLY,
                     attn_count * sizeof(int8_t), attn_wv_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_attn_wv_s = CreateKernelArgBuffer(
                     context, kernel_decode, 11, CL_MEM_READ_ONLY,
                     attn_s_count * sizeof(float), attn_wv_s_host.data(),
                     &err));
  OCL_CHECK(err, cl::Buffer buffer_attn_wo = CreateKernelArgBuffer(
                     context, kernel_decode, 12, CL_MEM_READ_ONLY,
                     attn_count * sizeof(int8_t), attn_wo_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_attn_wo_s = CreateKernelArgBuffer(
                     context, kernel_decode, 13, CL_MEM_READ_ONLY,
                     attn_s_count * sizeof(float), attn_wo_s_host.data(),
                     &err));
  OCL_CHECK(err, cl::Buffer buffer_rms_ffn = CreateKernelArgBuffer(
                     context, kernel_decode, 14, CL_MEM_READ_ONLY,
                     rms_count * sizeof(float), rms_ffn_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_ffn_w1 = CreateKernelArgBuffer(
                     context, kernel_decode, 15, CL_MEM_READ_ONLY,
                     ffn_a_count * sizeof(int8_t), ffn_w1_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_ffn_w1_s = CreateKernelArgBuffer(
                     context, kernel_decode, 16, CL_MEM_READ_ONLY,
                     ffn_a_s_count * sizeof(float), ffn_w1_s_host.data(),
                     &err));
  OCL_CHECK(err, cl::Buffer buffer_ffn_w2 = CreateKernelArgBuffer(
                     context, kernel_decode, 17, CL_MEM_READ_ONLY,
                     ffn_b_count * sizeof(int8_t), ffn_w2_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_ffn_w2_s = CreateKernelArgBuffer(
                     context, kernel_decode, 18, CL_MEM_READ_ONLY,
                     ffn_b_s_count * sizeof(float), ffn_w2_s_host.data(),
                     &err));
  OCL_CHECK(err, cl::Buffer buffer_ffn_w3 = CreateKernelArgBuffer(
                     context, kernel_decode, 19, CL_MEM_READ_ONLY,
                     ffn_a_count * sizeof(int8_t), ffn_w3_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_ffn_w3_s = CreateKernelArgBuffer(
                     context, kernel_decode, 20, CL_MEM_READ_ONLY,
                     ffn_a_s_count * sizeof(float), ffn_w3_s_host.data(),
                     &err));
  OCL_CHECK(err, cl::Buffer buffer_rms_final = CreateKernelArgBuffer(
                     context, kernel_decode, 21, CL_MEM_READ_ONLY,
                     rms_final_count * sizeof(float), rms_final_host.data(),
                     &err));
  OCL_CHECK(err, cl::Buffer buffer_cos = CreateKernelArgBuffer(
                     context, kernel_decode, 22, CL_MEM_READ_ONLY,
                     sincos_count * sizeof(float), cos_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_sin = CreateKernelArgBuffer(
                     context, kernel_decode, 23, CL_MEM_READ_ONLY,
                     sincos_count * sizeof(float), sin_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_k_cache = CreateKernelArgBuffer(
                     context, kernel_decode, 24, CL_MEM_READ_WRITE,
                     cache_count * sizeof(float), k_cache_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_v_cache = CreateKernelArgBuffer(
                     context, kernel_decode, 25, CL_MEM_READ_WRITE,
                     cache_count * sizeof(float), v_cache_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_next = CreateKernelArgBuffer(
                     context, kernel_decode, 26, CL_MEM_WRITE_ONLY,
                     sizeof(uint32_t), next_host.data(), &err));

  OCL_CHECK(err, err = kernel_decode.setArg(2, buffer_tok_emb));
  OCL_CHECK(err, err = kernel_decode.setArg(3, buffer_tok_emb_q));
  OCL_CHECK(err, err = kernel_decode.setArg(4, buffer_tok_emb_s));
  OCL_CHECK(err, err = kernel_decode.setArg(5, buffer_rms_att));
  OCL_CHECK(err, err = kernel_decode.setArg(6, buffer_attn_wq));
  OCL_CHECK(err, err = kernel_decode.setArg(7, buffer_attn_wq_s));
  OCL_CHECK(err, err = kernel_decode.setArg(8, buffer_attn_wk));
  OCL_CHECK(err, err = kernel_decode.setArg(9, buffer_attn_wk_s));
  OCL_CHECK(err, err = kernel_decode.setArg(10, buffer_attn_wv));
  OCL_CHECK(err, err = kernel_decode.setArg(11, buffer_attn_wv_s));
  OCL_CHECK(err, err = kernel_decode.setArg(12, buffer_attn_wo));
  OCL_CHECK(err, err = kernel_decode.setArg(13, buffer_attn_wo_s));
  OCL_CHECK(err, err = kernel_decode.setArg(14, buffer_rms_ffn));
  OCL_CHECK(err, err = kernel_decode.setArg(15, buffer_ffn_w1));
  OCL_CHECK(err, err = kernel_decode.setArg(16, buffer_ffn_w1_s));
  OCL_CHECK(err, err = kernel_decode.setArg(17, buffer_ffn_w2));
  OCL_CHECK(err, err = kernel_decode.setArg(18, buffer_ffn_w2_s));
  OCL_CHECK(err, err = kernel_decode.setArg(19, buffer_ffn_w3));
  OCL_CHECK(err, err = kernel_decode.setArg(20, buffer_ffn_w3_s));
  OCL_CHECK(err, err = kernel_decode.setArg(21, buffer_rms_final));
  OCL_CHECK(err, err = kernel_decode.setArg(22, buffer_cos));
  OCL_CHECK(err, err = kernel_decode.setArg(23, buffer_sin));
  OCL_CHECK(err, err = kernel_decode.setArg(24, buffer_k_cache));
  OCL_CHECK(err, err = kernel_decode.setArg(25, buffer_v_cache));
  OCL_CHECK(err, err = kernel_decode.setArg(26, buffer_next));

  OCL_CHECK(err,
            err = q.enqueueMigrateMemObjects(
                {buffer_tok_emb, buffer_tok_emb_q, buffer_tok_emb_s,
                 buffer_rms_att, buffer_attn_wq, buffer_attn_wq_s,
                 buffer_attn_wk, buffer_attn_wk_s, buffer_attn_wv,
                 buffer_attn_wv_s, buffer_attn_wo, buffer_attn_wo_s,
                 buffer_rms_ffn, buffer_ffn_w1, buffer_ffn_w1_s,
                 buffer_ffn_w2, buffer_ffn_w2_s, buffer_ffn_w3,
                 buffer_ffn_w3_s, buffer_rms_final, buffer_cos, buffer_sin,
                 buffer_k_cache, buffer_v_cache},
                0));
  OCL_CHECK(err, err = q.finish());
#endif // USE_CPU_ONLY

  // 6. Decode
  static llama2::Context ctx;
  llama2::Tensor1d ctx_input;
  static llama2::Tensor3dCache ctx_k_cache;
  static llama2::Tensor3dCache ctx_v_cache;
  llama2::Tensor1dLogits ctx_logits;
  llama2::Tensor1d ctx_final_norm;

  auto start_clk = std::chrono::steady_clock::now();

  int next;
  int token = 1; // BOS (Begin of Sequence)

#ifndef USE_CPU_ONLY
  if (args.temp >= 1e-5) {
    std::cerr << "[ERROR] FPGA build returns exact argmax token only; use "
                 "--temp 0 for now."
              << std::endl;
    return EXIT_FAILURE;
  }
#endif // USE_CPU_ONLY

  for (int pos = 0; pos < args.max_seq; ++pos) {

    // 6-1. Load the context input and decode the next token.
    llama2::CopyTensor1d(ctx_input, tok_emb_table[token]);
    llama2::Decode(token, pos, ctx_input, ctx_k_cache, ctx_v_cache,
                 ctx_final_norm, ctx_logits, next, weights
#ifndef USE_CPU_ONLY
                 ,
                 q, kernel_decode, next_host.data(), buffer_next
#endif // USE_CPU_ONLY
    );

#ifdef USE_CPU_ONLY
    // 6-2. Calculate the logits and softmax.
    llama2::MutmulVocab(ctx_logits, ctx_final_norm, weights.tok_emb_q,
                        weights.tok_emb_s);
#endif // USE_CPU_ONLY

    if (args.print_softmax) {
#ifdef USE_CPU_ONLY
      printf("\nSoftmax\n <- ");
      for (int i = 0; i <= pos; ++i)
        printf("%5.4f, ", ctx.attn_qk[0][i]);
      printf("\n -> ");
      for (int i = 0; i <= pos; ++i)
        printf("%5.4f, ", ctx.attn_sm[0][i]);
      printf("\n");
#else
      std::cerr << "--print_softmax is only available in CPU-only builds.\n";
#endif
    }

    // 6-3. Sampling the next token.
#ifdef USE_CPU_ONLY
    if (args.temp < 1e-5) {
      next = llama2::Argmax(ctx_logits);
    } else {
      for (int q = 0; q < vocab_size; ++q) {
        ctx_logits[q] /= args.temp;
      }
      llama2::Softmax(ctx_logits, ctx_logits);
      next = SelectFromLogits(ctx_logits);
    }
#endif // USE_CPU_ONLY

    const std::string piece = llama2::DecodePiece(vocab, token, next);
    if (args.color) {
      std::cout << "\e[31m";
      std::cout.write(piece.data(), piece.size());
      std::cout << "\e[0m";
    } else {
      std::cout.write(piece.data(), piece.size());
    }
    std::cout << std::flush;

    // Dump the contexts.
    if (args.log) {
#ifdef USE_CPU_ONLY
      DumpContext("log/" + std::to_string(pos) + "_", ctx, llama2::kNumLayers);
#else
      std::cerr << "--log is only available in CPU-only builds.\n";
#endif
    }

    token = next;
  }
  std::cout << "\n";

  // 7. Print the time and speed.
  auto end_clk = std::chrono::steady_clock::now();
  double decode_time =
      std::chrono::duration<double>(end_clk - start_clk).count();
  std::cout << "Time : " << decode_time << "[s]" << std::endl
            << "Speed: " << args.max_seq / decode_time << "[tok/s]"
            << std::endl;

#ifndef USE_CPU_ONLY
  // 8. Flush OpenCL commands.
  OCL_CHECK(err, err = q.finish());
  delete[] buf;
#endif // USE_CPU_ONLY

  return 0;
}
