#include "vocab.hpp"

#include <cstdio>
#include <stdexcept>

namespace swan {

// ResizeVocab resizes the vocab to the given size.
void ResizeVocab(Vocab& vocab, int vocab_size) {
  vocab.dict.resize(vocab_size);
  for (size_t i = 0; i < vocab.byte_pieces.size(); i++) {
    vocab.byte_pieces.at(i) = std::string(1, static_cast<char>(i));
  }
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
    vocab.dict.at(i) = std::move(piece);
  }
}

std::string DecodePiece(const Vocab& vocab, int prev_token, int token) {
  std::string piece = vocab.dict.at(token);
  if (prev_token == 1 && !piece.empty() && piece.front() == ' ') {
    piece.erase(piece.begin());
  }

  unsigned int byte_val;
  if (std::sscanf(piece.c_str(), "<0x%02X>", &byte_val) == 1 &&
      byte_val < vocab.byte_pieces.size()) {
    return vocab.byte_pieces.at(byte_val);
  }

  return piece;
}

} // namespace swan
