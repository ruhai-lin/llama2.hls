#include "vocab.hpp"

#include <stdexcept>

namespace llama2 {

// ResizeVocab resizes the vocab to the given size.
void ResizeVocab(Vocab& vocab, int vocab_size) {
  vocab.dict.resize(vocab_size);
}

// LoadVocab loads llama2.c tokenizer.bin:
// int max_token_length, then repeated {float score, int len, bytes token}.
void LoadVocab(Vocab& vocab, std::ifstream& fs) {
  int max_token_length;
  fs.read(reinterpret_cast<char*>(&max_token_length), sizeof(int));
  if (!fs || max_token_length <= 0) {
    throw std::runtime_error("failed to read tokenizer max token length");
  }

  for (size_t i = 0; i < vocab.dict.size(); i++) {
    float score;
    int len;
    fs.read(reinterpret_cast<char*>(&score), sizeof(float));
    fs.read(reinterpret_cast<char*>(&len), sizeof(int));
    if (!fs || len < 0 || len > max_token_length * 4) {
      throw std::runtime_error("failed to read tokenizer token metadata");
    }

    std::string piece(len, '\0');
    fs.read(piece.data(), len);
    if (!fs) {
      throw std::runtime_error("failed to read tokenizer token bytes");
    }
    piece.push_back('\0');
    vocab.dict.at(i) = std::move(piece);
  }
}

} // namespace llama2
