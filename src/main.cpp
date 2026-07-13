#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "context.hpp"
#include "decode.hpp"
#include "vocab.hpp"
#include "weight.hpp"

#ifndef USE_CPU_ONLY
#include <stdlib.h>

#include <httplib.h>
#include <nlohmann/json.hpp>

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/opencl.hpp>

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
  void deallocate(T* p, std::size_t) { free(p); }
};

template <typename Scalar, typename T>
std::vector<Scalar, aligned_allocator<Scalar>> FlattenWeights(const T& data,
                                                              std::size_t count) {
  std::vector<Scalar, aligned_allocator<Scalar>> flat(count);
  std::memcpy(flat.data(), reinterpret_cast<const Scalar*>(&data),
              count * sizeof(Scalar));
  return flat;
}

template <std::size_t Rows, std::size_t Cols, std::size_t Groups>
void AppendPackedMatrix(
    std::vector<llama2::PackedParameterWord,
                aligned_allocator<llama2::PackedParameterWord>>& packed,
    const int8_t (&weight)[Rows][Cols], const float (&scales)[Rows][Groups]) {
  static_assert(Rows % llama2::kPackedRows == 0);
  static_assert(Cols == Groups * llama2::kQuantGroupSize);

  for (std::size_t row_base = 0; row_base < Rows;
       row_base += llama2::kPackedRows) {
    for (std::size_t group = 0; group < Groups; ++group) {
      llama2::PackedParameterWord scale_word{};
      for (int row = 0; row < llama2::kPackedRows; ++row) {
        std::memcpy(&scale_word[row * sizeof(float)],
                    &scales[row_base + row][group], sizeof(float));
      }
      packed.push_back(scale_word);

      for (int pair = 0; pair < llama2::kPackedRows / 2; ++pair) {
        llama2::PackedParameterWord weight_word{};
        const std::size_t row0 = row_base + pair * 2;
        const std::size_t row1 = row0 + 1;
        const std::size_t col = group * llama2::kQuantGroupSize;
        std::memcpy(&weight_word[0], &weight[row0][col],
                    llama2::kQuantGroupSize);
        std::memcpy(&weight_word[llama2::kQuantGroupSize],
                    &weight[row1][col], llama2::kQuantGroupSize);
        packed.push_back(weight_word);
      }
    }
  }
}

template <std::size_t Layers, std::size_t Rows, std::size_t Cols,
          std::size_t Groups>
void AppendPackedMatrix(
    std::vector<llama2::PackedParameterWord,
                aligned_allocator<llama2::PackedParameterWord>>& packed,
    const int8_t (&weight)[Layers][Rows][Cols],
    const float (&scales)[Layers][Rows][Groups]) {
  for (std::size_t layer = 0; layer < Layers; ++layer) {
    AppendPackedMatrix(packed, weight[layer], scales[layer]);
  }
}

auto PackParameters(const llama2::Weights& weights) {
  std::vector<llama2::PackedParameterWord,
              aligned_allocator<llama2::PackedParameterWord>>
      packed;
  packed.reserve(llama2::kPackedParameterWords);
  AppendPackedMatrix(packed, weights.tok_emb_q, weights.tok_emb_s);
  AppendPackedMatrix(packed, weights.attn_wq, weights.attn_wq_s);
  AppendPackedMatrix(packed, weights.attn_wk, weights.attn_wk_s);
  AppendPackedMatrix(packed, weights.attn_wv, weights.attn_wv_s);
  AppendPackedMatrix(packed, weights.attn_wo, weights.attn_wo_s);
  AppendPackedMatrix(packed, weights.ffn_w1, weights.ffn_w1_s);
  AppendPackedMatrix(packed, weights.ffn_w2, weights.ffn_w2_s);
  AppendPackedMatrix(packed, weights.ffn_w3, weights.ffn_w3_s);
  if (packed.size() != llama2::kPackedParameterWords) {
    throw std::runtime_error("packed parameter size mismatch");
  }
  return packed;
}

template <typename T>
cl::Buffer CreateKernelArgBuffer(const cl::Context& context,
                                 const cl::Kernel& kernel, unsigned int argidx,
                                 cl_mem_flags flags, std::size_t bytes,
                                 T* host_ptr, cl_int* err) {
  (void)kernel;
  (void)argidx;
  (void)host_ptr;
  return cl::Buffer(context, flags, bytes, nullptr, err);
}
#endif // USE_CPU_ONLY

// Command line arguments.
struct Args {
  std::string weight_path = "./model/stories15M_q8.bin";
  std::string vocab_path = "./model/tokenizer.bin";
  std::string mode = "generate";
  std::string prompt;
  std::string system_prompt;
  std::string host = "127.0.0.1";
  uint64_t max_seq = 256;
  float temp = 0.0;
  int port = 8080;
  bool has_system_prompt = false;
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
    } else if ((std::strcmp(argv[i], "-m") == 0 ||
                std::strcmp(argv[i], "--m") == 0 ||
                std::strcmp(argv[i], "--mode") == 0) &&
               i + 1 < argc) {
      args.mode = argv[++i];
    } else if ((std::strcmp(argv[i], "-i") == 0 ||
                std::strcmp(argv[i], "--prompt") == 0) &&
               i + 1 < argc) {
      args.prompt = argv[++i];
    } else if ((std::strcmp(argv[i], "-y") == 0 ||
                std::strcmp(argv[i], "--system_prompt") == 0) &&
               i + 1 < argc) {
      args.system_prompt = argv[++i];
      args.has_system_prompt = true;
    } else if (std::strcmp(argv[i], "--max_seq") == 0 && i + 1 < argc) {
      args.max_seq = std::stoull(argv[++i]);
    } else if (std::strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
      args.temp = std::stof(argv[++i]);
    } else if (std::strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
      args.host = argv[++i];
    } else if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
      args.port = std::stoi(argv[++i]);
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

constexpr std::string_view kModelId = "tinystories-15m-w8a8-kv260";
using Forward = std::function<int(int, int)>;

class Sequence {
public:
  explicit Sequence(const Forward& forward) : forward_(forward) {}

  int Advance(int token) {
    if (pos_ >= llama2::kSeqLen) {
      throw std::runtime_error("model context is full");
    }
    return forward_(token, pos_++);
  }

  int pos() const { return pos_; }

private:
  const Forward& forward_;
  int pos_ = 0;
};

struct CompletionResult {
  std::string text;
  int prompt_tokens = 0;
  int completion_tokens = 0;
  int forward_tokens = 0;
  bool stopped = false;
};

CompletionResult Complete(const llama2::Vocab& vocab, std::string_view prompt,
                          int max_tokens, const Forward& forward,
                          const std::function<void(std::string_view)>&
                              on_piece = {}) {
  if (max_tokens <= 0) {
    throw std::invalid_argument("max_tokens must be positive");
  }

  const std::vector<int> prompt_tokens =
      llama2::Encode(vocab, prompt, true, false);
  if (prompt_tokens.empty()) {
    throw std::runtime_error("tokenizer produced an empty prompt");
  }
  if (prompt_tokens.size() + static_cast<std::size_t>(max_tokens) - 1 >
      static_cast<std::size_t>(llama2::kSeqLen)) {
    throw std::invalid_argument("prompt and completion exceed model context");
  }

  Sequence sequence(forward);
  int next = 0;
  for (const int token : prompt_tokens) {
    next = sequence.Advance(token);
  }

  CompletionResult result;
  result.prompt_tokens = static_cast<int>(prompt_tokens.size());
  int previous = prompt_tokens.back();
  for (int i = 0; i < max_tokens; ++i) {
    if (next == 1) {
      result.stopped = true;
      break;
    }

    const std::string piece = llama2::DecodePiece(vocab, previous, next);
    result.text += piece;
    ++result.completion_tokens;
    if (on_piece) {
      on_piece(piece);
    }

    previous = next;
    if (i + 1 < max_tokens) {
      next = sequence.Advance(next);
    }
  }
  result.forward_tokens = sequence.pos();
  return result;
}

void RunChat(const Args& args, const llama2::Vocab& vocab,
             const Forward& forward) {
  Sequence sequence(forward);
  std::string system_prompt = args.system_prompt;
  bool first_turn = true;

  if (!args.has_system_prompt) {
    std::cout << "Enter system prompt (optional): " << std::flush;
    if (!std::getline(std::cin, system_prompt)) {
      return;
    }
  }

  while (sequence.pos() < static_cast<int>(args.max_seq)) {
    std::string user_prompt;
    if (first_turn && !args.prompt.empty()) {
      user_prompt = args.prompt;
    } else {
      std::cout << "User: " << std::flush;
      if (!std::getline(std::cin, user_prompt)) {
        break;
      }
    }

    std::ostringstream rendered;
    if (first_turn && !system_prompt.empty()) {
      rendered << "[INST] <<SYS>>\n"
               << system_prompt << "\n<</SYS>>\n\n"
               << user_prompt << " [/INST]";
    } else {
      rendered << "[INST] " << user_prompt << " [/INST]";
    }
    first_turn = false;

    const std::vector<int> prompt_tokens =
        llama2::Encode(vocab, rendered.str(), true, false);
    if (sequence.pos() + static_cast<int>(prompt_tokens.size()) >
        static_cast<int>(args.max_seq)) {
      std::cout << "Context is full.\n";
      break;
    }

    std::cout << "Assistant: " << std::flush;
    int next = 0;
    for (const int token : prompt_tokens) {
      next = sequence.Advance(token);
    }

    int previous = prompt_tokens.back();
    while (next != 2) {
      const std::string piece = llama2::DecodePiece(vocab, previous, next);
      std::cout.write(piece.data(), piece.size());
      std::cout << std::flush;
      previous = next;
      if (sequence.pos() >= static_cast<int>(args.max_seq)) {
        break;
      }
      next = sequence.Advance(next);
    }
    std::cout << "\n";

    if (next == 2 && sequence.pos() < static_cast<int>(args.max_seq)) {
      (void)sequence.Advance(next);
    }
  }
}

#ifndef USE_CPU_ONLY
using Json = nlohmann::json;

void SetJson(httplib::Response& response, const Json& body,
             int status = 200) {
  response.status = status;
  response.set_content(body.dump(), "application/json");
}

Json ErrorJson(int code, std::string_view message,
               std::string_view type = "invalid_request_error") {
  return {{"error", {{"code", code}, {"message", message}, {"type", type}}}};
}

void RunServer(const Args& args, const llama2::Vocab& vocab,
               const Forward& forward) {
  httplib::Server server;
  server.new_task_queue = [] { return new httplib::ThreadPool(1); };
  uint64_t completion_id = 0;

  server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    SetJson(res, {{"status", "ok"}});
  });

  server.Get("/v1/models",
             [](const httplib::Request&, httplib::Response& res) {
               SetJson(res,
                       {{"object", "list"},
                        {"data",
                         {{{"id", kModelId},
                           {"object", "model"},
                           {"owned_by", "llama2.hls"},
                           {"meta",
                            {{"backend", "XRT"},
                             {"device", "KV260"},
                             {"context_length", llama2::kSeqLen},
                             {"temperature", 0}}}}}}});
             });

  server.Post("/v1/completions",
              [&](const httplib::Request& request, httplib::Response& res) {
                try {
                  const Json input = Json::parse(request.body);
                  if (!input.is_object()) {
                    throw std::invalid_argument("request body must be an object");
                  }
                  if (!input.contains("model") ||
                      !input.at("model").is_string() ||
                      input.at("model").get<std::string>() != kModelId) {
                    throw std::invalid_argument("unknown model");
                  }
                  if (!input.contains("prompt") ||
                      !input.at("prompt").is_string()) {
                    throw std::invalid_argument("prompt must be a string");
                  }
                  if (input.value("stream", false)) {
                    throw std::invalid_argument(
                        "streaming is not supported in this version");
                  }
                  const double temperature = input.value("temperature", 0.0);
                  if (temperature != 0.0) {
                    throw std::invalid_argument("temperature must be 0");
                  }
                  const int max_tokens = input.value("max_tokens", 32);
                  const CompletionResult result = Complete(
                      vocab, input.at("prompt").get<std::string>(), max_tokens,
                      forward);

                  SetJson(
                      res,
                      {{"id", "cmpl-kv260-" +
                                  std::to_string(++completion_id)},
                       {"object", "text_completion"},
                       {"model", kModelId},
                       {"choices",
                        {{{"text", result.text},
                          {"index", 0},
                          {"logprobs", nullptr},
                          {"finish_reason",
                           result.stopped ? "stop" : "length"}}}},
                       {"usage",
                        {{"prompt_tokens", result.prompt_tokens},
                         {"completion_tokens", result.completion_tokens},
                         {"total_tokens", result.prompt_tokens +
                                              result.completion_tokens}}}});
                } catch (const nlohmann::json::exception& error) {
                  SetJson(res, ErrorJson(400, error.what()), 400);
                } catch (const std::exception& error) {
                  SetJson(res, ErrorJson(400, error.what()), 400);
                }
              });

  std::cout << "Listening on http://" << args.host << ':' << args.port
            << std::endl;
  if (!server.listen(args.host, args.port)) {
    throw std::runtime_error("failed to start HTTP server");
  }
}
#endif // USE_CPU_ONLY

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
              << "  -m, --mode      : generate|chat|server" << std::endl
              << "  -i, --prompt    : Input prompt in generate mode" << std::endl
              << "  -y, --system_prompt: Initial system prompt in chat mode"
              << std::endl
              << "  --max_seq       : Maximum sequence length" << std::endl
              << "  --temp          : Temperature for sampling" << std::endl
              << "  --host          : Server bind address" << std::endl
              << "  --port          : Server port" << std::endl
              << "  --color         : Enable color output" << std::endl
              << "  --log           : Enable log output" << std::endl
              << "  --help, -h      : Show this help message" << std::endl;
    return 0;
  }
  if (args.max_seq > llama2::kSeqLen) {
    std::cerr << "[ERROR] --max_seq exceeds model sequence length "
              << llama2::kSeqLen << std::endl;
    return EXIT_FAILURE;
  }
  if (args.mode != "generate" && args.mode != "chat" &&
      args.mode != "server") {
    std::cerr << "[ERROR] Unknown mode: " << args.mode << std::endl;
    return EXIT_FAILURE;
  }
  if (args.port <= 0 || args.port > 65535) {
    std::cerr << "[ERROR] --port must be in [1, 65535]" << std::endl;
    return EXIT_FAILURE;
  }
#ifdef USE_CPU_ONLY
  if (args.mode == "server") {
    std::cerr << "[ERROR] server mode is only available in the FPGA host build"
              << std::endl;
    return EXIT_FAILURE;
  }
#endif

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
  constexpr std::size_t rms_count = llama2::kNumLayers * llama2::kDim;
  constexpr std::size_t rms_final_count = llama2::kDim;
  constexpr std::size_t sincos_count = llama2::kSeqLen * llama2::kSinCosTable;
  constexpr std::size_t cache_count =
      llama2::kNumLayers * llama2::kSeqLen * llama2::kDim;

  auto tok_emb_host = FlattenWeights<float>(tok_emb_table, tok_emb_count);
  auto packed_params_host = PackParameters(weights);
  auto rms_att_host = FlattenWeights<float>(weights.rms_att_w, rms_count);
  auto rms_ffn_host = FlattenWeights<float>(weights.rms_ffn_w, rms_count);
  auto rms_final_host =
      FlattenWeights<float>(weights.rms_final, rms_final_count);
  auto cos_host = FlattenWeights<float>(weights.cos_table, sincos_count);
  auto sin_host = FlattenWeights<float>(weights.sin_table, sincos_count);
  std::vector<float, aligned_allocator<float>> k_cache_host(cache_count, 0.0f);
  std::vector<float, aligned_allocator<float>> v_cache_host(cache_count, 0.0f);
  std::vector<uint32_t, aligned_allocator<uint32_t>> next_host(1, 0);

  std::vector<cl::Device> devices;
  cl_int err;
  cl::Context context;
  cl::CommandQueue q;
  cl::Kernel decode_kernel;
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
      OCL_CHECK(err, decode_kernel = cl::Kernel(program, "decode", &err));
      std::cout << "load : decode" << std::endl;
      valid_device = true;
      break; // we break because we found a valid device
    }
  }
  if (!valid_device) {
    std::cout << "Failed to program any device found, exit!\n";
    exit(EXIT_FAILURE);
  }

  OCL_CHECK(err, cl::Buffer buffer_tok_emb = CreateKernelArgBuffer(
                     context, decode_kernel, 2, CL_MEM_READ_ONLY,
                     tok_emb_count * sizeof(float), tok_emb_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_packed_params = CreateKernelArgBuffer(
                     context, decode_kernel, 3, CL_MEM_READ_ONLY,
                     packed_params_host.size() *
                         sizeof(llama2::PackedParameterWord),
                     packed_params_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_rms_att = CreateKernelArgBuffer(
                     context, decode_kernel, 4, CL_MEM_READ_ONLY,
                     rms_count * sizeof(float), rms_att_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_rms_ffn = CreateKernelArgBuffer(
                     context, decode_kernel, 5, CL_MEM_READ_ONLY,
                     rms_count * sizeof(float), rms_ffn_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_rms_final = CreateKernelArgBuffer(
                     context, decode_kernel, 6, CL_MEM_READ_ONLY,
                     rms_final_count * sizeof(float), rms_final_host.data(),
                     &err));
  OCL_CHECK(err, cl::Buffer buffer_cos = CreateKernelArgBuffer(
                     context, decode_kernel, 7, CL_MEM_READ_ONLY,
                     sincos_count * sizeof(float), cos_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_sin = CreateKernelArgBuffer(
                     context, decode_kernel, 8, CL_MEM_READ_ONLY,
                     sincos_count * sizeof(float), sin_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_k_cache = CreateKernelArgBuffer(
                     context, decode_kernel, 9, CL_MEM_READ_WRITE,
                     cache_count * sizeof(float), k_cache_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_v_cache = CreateKernelArgBuffer(
                     context, decode_kernel, 10, CL_MEM_READ_WRITE,
                     cache_count * sizeof(float), v_cache_host.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_next = CreateKernelArgBuffer(
                     context, decode_kernel, 11, CL_MEM_WRITE_ONLY,
                     sizeof(uint32_t), next_host.data(), &err));

  OCL_CHECK(err, err = decode_kernel.setArg(2, buffer_tok_emb));
  OCL_CHECK(err, err = decode_kernel.setArg(3, buffer_packed_params));
  OCL_CHECK(err, err = decode_kernel.setArg(4, buffer_rms_att));
  OCL_CHECK(err, err = decode_kernel.setArg(5, buffer_rms_ffn));
  OCL_CHECK(err, err = decode_kernel.setArg(6, buffer_rms_final));
  OCL_CHECK(err, err = decode_kernel.setArg(7, buffer_cos));
  OCL_CHECK(err, err = decode_kernel.setArg(8, buffer_sin));
  OCL_CHECK(err, err = decode_kernel.setArg(9, buffer_k_cache));
  OCL_CHECK(err, err = decode_kernel.setArg(10, buffer_v_cache));
  OCL_CHECK(err, err = decode_kernel.setArg(11, buffer_next));

  OCL_CHECK(err, err = q.enqueueWriteBuffer(buffer_tok_emb, CL_FALSE, 0,
                                            tok_emb_count * sizeof(float),
                                            tok_emb_host.data()));
  OCL_CHECK(err, err = q.enqueueWriteBuffer(
                     buffer_packed_params, CL_FALSE, 0,
                     packed_params_host.size() *
                         sizeof(llama2::PackedParameterWord),
                     packed_params_host.data()));
  OCL_CHECK(err, err = q.enqueueWriteBuffer(buffer_rms_att, CL_FALSE, 0,
                                            rms_count * sizeof(float),
                                            rms_att_host.data()));
  OCL_CHECK(err, err = q.enqueueWriteBuffer(buffer_rms_ffn, CL_FALSE, 0,
                                            rms_count * sizeof(float),
                                            rms_ffn_host.data()));
  OCL_CHECK(err, err = q.enqueueWriteBuffer(buffer_rms_final, CL_FALSE, 0,
                                            rms_final_count * sizeof(float),
                                            rms_final_host.data()));
  OCL_CHECK(err, err = q.enqueueWriteBuffer(buffer_cos, CL_FALSE, 0,
                                            sincos_count * sizeof(float),
                                            cos_host.data()));
  OCL_CHECK(err, err = q.enqueueWriteBuffer(buffer_sin, CL_FALSE, 0,
                                            sincos_count * sizeof(float),
                                            sin_host.data()));
  OCL_CHECK(err, err = q.enqueueWriteBuffer(buffer_k_cache, CL_FALSE, 0,
                                            cache_count * sizeof(float),
                                            k_cache_host.data()));
  OCL_CHECK(err, err = q.enqueueWriteBuffer(buffer_v_cache, CL_FALSE, 0,
                                            cache_count * sizeof(float),
                                            v_cache_host.data()));
  OCL_CHECK(err, err = q.finish());
#endif // USE_CPU_ONLY

  // 6. Decode
#ifdef USE_CPU_ONLY
  static llama2::Context ctx;
#endif
  llama2::Tensor1d ctx_input;
  static llama2::Tensor3dCache ctx_k_cache;
  static llama2::Tensor3dCache ctx_v_cache;
  llama2::Tensor1dLogits ctx_logits;
  llama2::Tensor1d ctx_final_norm;

#ifndef USE_CPU_ONLY
  if (args.temp >= 1e-5) {
    std::cerr << "[ERROR] FPGA build returns exact argmax token only; use "
                 "--temp 0 for now."
              << std::endl;
    return EXIT_FAILURE;
  }
#endif // USE_CPU_ONLY

  const Forward forward = [&](int token, int pos) {
    int next = 0;
    llama2::CopyTensor1d(ctx_input, tok_emb_table[token]);
    llama2::Decode(token, pos, ctx_input, ctx_k_cache, ctx_v_cache,
                   ctx_final_norm, ctx_logits, next, weights
#ifndef USE_CPU_ONLY
                   ,
                   q, decode_kernel, next_host.data(), buffer_next
#endif // USE_CPU_ONLY
    );

#ifdef USE_CPU_ONLY
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

    if (args.log) {
#ifdef USE_CPU_ONLY
      DumpContext("log/" + std::to_string(pos) + "_", ctx, llama2::kNumLayers);
#else
      std::cerr << "--log is only available in CPU-only builds.\n";
#endif
    }
    return next;
  };

  if (args.mode == "chat") {
    RunChat(args, vocab, forward);
  } else if (args.mode == "server") {
#ifndef USE_CPU_ONLY
    RunServer(args, vocab, forward);
#endif
  } else {
    const std::vector<int> prompt_tokens =
        llama2::Encode(vocab, args.prompt, true, false);
    const int max_tokens = static_cast<int>(args.max_seq) -
                           static_cast<int>(prompt_tokens.size()) + 1;
    if (max_tokens <= 0) {
      std::cerr << "[ERROR] prompt exceeds --max_seq" << std::endl;
      return EXIT_FAILURE;
    }

    const auto start_clk = std::chrono::steady_clock::now();
    const auto print_piece = [&](std::string_view piece) {
      if (args.color) {
        std::cout << "\e[31m";
        std::cout.write(piece.data(), piece.size());
        std::cout << "\e[0m";
      } else {
        std::cout.write(piece.data(), piece.size());
      }
      std::cout << std::flush;
    };
    const CompletionResult result =
        Complete(vocab, args.prompt, max_tokens, forward, print_piece);
    std::cout << "\n";

    const auto end_clk = std::chrono::steady_clock::now();
    const double decode_time =
        std::chrono::duration<double>(end_clk - start_clk).count();
    std::cout << "Time : " << decode_time << "[s]" << std::endl
              << "Speed: " << result.forward_tokens / decode_time << "[tok/s]"
              << std::endl;
  }

#ifndef USE_CPU_ONLY
  OCL_CHECK(err, err = q.finish());
  delete[] buf;
  std::cout.flush();
  std::cerr.flush();
  std::_Exit(EXIT_SUCCESS);
#endif // USE_CPU_ONLY

  return 0;
}
