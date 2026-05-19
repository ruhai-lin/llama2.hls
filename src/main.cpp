#include <cstring>
#include <chrono>
#include <iomanip>
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

template <typename T>
std::vector<float, aligned_allocator<float>> FlattenWeights(const T& data,
                                                            std::size_t count) {
  std::vector<float, aligned_allocator<float>> flat(count);
  std::memcpy(flat.data(), reinterpret_cast<const float*>(&data),
              count * sizeof(float));
  return flat;
}
#endif // USE_CPU_ONLY

// Command line arguments.
struct Args {
  std::string weight_path = "./model/stories15M.bin";
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
int SelectFromLogits(const swan::Tensor1dLogits& prob_dist) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);

  const int vocab_size = swan::kVocabSize;
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
            << "  dim       : " << swan::kDim << std::endl
            << "  ffn_dim   : " << swan::kFFNDim << std::endl
            << "  n_layers  : " << swan::kNumLayers << std::endl
            << "  n_heads   : " << swan::kNumHeads << std::endl
            << "  n_kv_heads: " << swan::kNumKVHeads << std::endl
            << "  vocab_size: " << swan::kVocabSize << std::endl
            << "  seq_len   : " << swan::kSeqLen << std::endl;

  // 3. Load model parameters.
  std::ifstream weight_fs(args.weight_path, std::ios::in | std::ios::binary);
  if (!weight_fs) {
    std::cout << "Failed to open: " << args.weight_path << std::endl;
    return EXIT_FAILURE;
  }
  static swan::Weights weights;
  static swan::Tensor2dTok tok_emb_table; // [vocab_size, dim]
  swan::LoadWeights(weights, tok_emb_table, weight_fs);
  weight_fs.close();

  // 4. Load vocabrary.
  std::ifstream vocab_fs(args.vocab_path, std::ios::in | std::ios::binary);
  if (!vocab_fs) {
    std::cout << "Failed to open: " << args.vocab_path << std::endl;
    return EXIT_FAILURE;
  }
  static swan::Vocab vocab;
  const int vocab_size = swan::kVocabSize;
  swan::ResizeVocab(vocab, vocab_size);
  swan::LoadVocab(vocab, vocab_fs);
  vocab_fs.close();

#ifndef USE_CPU_ONLY
  // 5. OpenCL Settings
  std::string xclbinFilename = "./binary_container_1.bin";

  constexpr std::size_t tok_emb_count = swan::kVocabSize * swan::kDim;
  constexpr std::size_t rms_count = swan::kNumLayers * swan::kDim;
  constexpr std::size_t attn_count =
      swan::kNumLayers * swan::kDim * swan::kDim;
  constexpr std::size_t ffn_a_count =
      swan::kNumLayers * swan::kFFNDim * swan::kDim;
  constexpr std::size_t ffn_b_count =
      swan::kNumLayers * swan::kDim * swan::kFFNDim;
  constexpr std::size_t rms_final_count = swan::kDim;
  constexpr std::size_t sincos_count = swan::kSeqLen * swan::kSinCosTable;
  constexpr std::size_t cache_count =
      swan::kNumLayers * swan::kSeqLen * swan::kDim;
  constexpr std::size_t logits_count = swan::kVocabSize;

  auto tok_emb_host = FlattenWeights(tok_emb_table, tok_emb_count);
  auto rms_att_host = FlattenWeights(weights.rms_att_w, rms_count);
  auto attn_wq_host = FlattenWeights(weights.attn_wq, attn_count);
  auto attn_wk_host = FlattenWeights(weights.attn_wk, attn_count);
  auto attn_wv_host = FlattenWeights(weights.attn_wv, attn_count);
  auto attn_wo_host = FlattenWeights(weights.attn_wo, attn_count);
  auto rms_ffn_host = FlattenWeights(weights.rms_ffn_w, rms_count);
  auto ffn_w1_host = FlattenWeights(weights.ffn_w1, ffn_a_count);
  auto ffn_w2_host = FlattenWeights(weights.ffn_w2, ffn_b_count);
  auto ffn_w3_host = FlattenWeights(weights.ffn_w3, ffn_a_count);
  auto rms_final_host = FlattenWeights(weights.rms_final, rms_final_count);
  auto cos_host = FlattenWeights(weights.cos_table, sincos_count);
  auto sin_host = FlattenWeights(weights.sin_table, sincos_count);
  std::vector<float, aligned_allocator<float>> k_cache_host(cache_count, 0.0f);
  std::vector<float, aligned_allocator<float>> v_cache_host(cache_count, 0.0f);
  std::vector<float, aligned_allocator<float>> logits_host(logits_count, 0.0f);

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

  OCL_CHECK(err, cl::Buffer buffer_tok_emb(
                     context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     tok_emb_count * sizeof(float), tok_emb_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_rms_att(
                     context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     rms_count * sizeof(float), rms_att_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_attn_wq(
                     context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     attn_count * sizeof(float), attn_wq_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_attn_wk(
                     context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     attn_count * sizeof(float), attn_wk_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_attn_wv(
                     context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     attn_count * sizeof(float), attn_wv_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_attn_wo(
                     context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     attn_count * sizeof(float), attn_wo_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_rms_ffn(
                     context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     rms_count * sizeof(float), rms_ffn_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_ffn_w1(
                     context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     ffn_a_count * sizeof(float), ffn_w1_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_ffn_w2(
                     context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     ffn_b_count * sizeof(float), ffn_w2_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_ffn_w3(
                     context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     ffn_a_count * sizeof(float), ffn_w3_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_rms_final(
                     context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     rms_final_count * sizeof(float), rms_final_host.data(),
                     &err));
  OCL_CHECK(err, cl::Buffer buffer_cos(
                     context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     sincos_count * sizeof(float), cos_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_sin(
                     context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     sincos_count * sizeof(float), sin_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_k_cache(
                     context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                     cache_count * sizeof(float), k_cache_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_v_cache(
                     context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                     cache_count * sizeof(float), v_cache_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_logits(
                     context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                     logits_count * sizeof(float), logits_host.data(), &err));

  OCL_CHECK(err, err = kernel_decode.setArg(2, buffer_tok_emb));
  OCL_CHECK(err, err = kernel_decode.setArg(3, buffer_rms_att));
  OCL_CHECK(err, err = kernel_decode.setArg(4, buffer_attn_wq));
  OCL_CHECK(err, err = kernel_decode.setArg(5, buffer_attn_wk));
  OCL_CHECK(err, err = kernel_decode.setArg(6, buffer_attn_wv));
  OCL_CHECK(err, err = kernel_decode.setArg(7, buffer_attn_wo));
  OCL_CHECK(err, err = kernel_decode.setArg(8, buffer_rms_ffn));
  OCL_CHECK(err, err = kernel_decode.setArg(9, buffer_ffn_w1));
  OCL_CHECK(err, err = kernel_decode.setArg(10, buffer_ffn_w2));
  OCL_CHECK(err, err = kernel_decode.setArg(11, buffer_ffn_w3));
  OCL_CHECK(err, err = kernel_decode.setArg(12, buffer_rms_final));
  OCL_CHECK(err, err = kernel_decode.setArg(13, buffer_cos));
  OCL_CHECK(err, err = kernel_decode.setArg(14, buffer_sin));
  OCL_CHECK(err, err = kernel_decode.setArg(15, buffer_k_cache));
  OCL_CHECK(err, err = kernel_decode.setArg(16, buffer_v_cache));
  OCL_CHECK(err, err = kernel_decode.setArg(17, buffer_logits));

  OCL_CHECK(err,
            err = q.enqueueMigrateMemObjects(
                {buffer_tok_emb, buffer_rms_att, buffer_attn_wq,
                 buffer_attn_wk, buffer_attn_wv, buffer_attn_wo,
                 buffer_rms_ffn, buffer_ffn_w1, buffer_ffn_w2, buffer_ffn_w3,
                 buffer_rms_final, buffer_cos, buffer_sin, buffer_k_cache,
                 buffer_v_cache},
                0));
  OCL_CHECK(err, err = q.finish());
#endif // USE_CPU_ONLY

  // 6. Decode
  static swan::Context ctx;
  swan::Tensor1d ctx_input;
  static swan::Tensor3dCache ctx_k_cache;
  static swan::Tensor3dCache ctx_v_cache;
  swan::Tensor1dLogits ctx_logits;
  swan::Tensor1d ctx_final_norm;

  auto start_clk = std::chrono::steady_clock::now();

  int next;
  int token = 1; // BOS (Begin of Sequence)

  for (int pos = 0; pos < args.max_seq; ++pos) {

    // 6-1. Load the context input and decode the next token.
    swan::CopyTensor1d(ctx_input, tok_emb_table[token]);
    swan::Decode(token, pos, ctx_input, ctx_k_cache, ctx_v_cache,
                 ctx_final_norm, ctx_logits, weights
#ifndef USE_CPU_ONLY
                 ,
                 q, kernel_decode, logits_host.data(), buffer_logits
#endif // USE_CPU_ONLY
    );

#ifdef USE_CPU_ONLY
    // 6-2. Calculate the logits and softmax.
    swan::MutmulVocab(ctx_logits, ctx_final_norm, tok_emb_table);
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
    if (args.temp < 1e-5) {
      next = swan::Argmax(ctx_logits);
    } else {
      for (int q = 0; q < vocab_size; ++q) {
        ctx_logits[q] /= args.temp;
      }
      swan::Softmax(ctx_logits, ctx_logits);
      next = SelectFromLogits(ctx_logits);
    }

    const std::string piece = swan::DecodePiece(vocab, token, next);
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
      DumpContext("log/" + std::to_string(pos) + "_", ctx, swan::kNumLayers);
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
