#include "vocab.hpp"

#include <cstdio>
#include <limits>
#include <stdexcept>

namespace llama2 {

// ResizeVocab resizes the vocab to the given size.
void ResizeVocab(Vocab& vocab, int vocab_size) {
  vocab.dict.resize(vocab_size);
  vocab.scores.resize(vocab_size);
  vocab.token_ids.clear();
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
    int len;
    fs.read(reinterpret_cast<char*>(&vocab.scores.at(i)), sizeof(float));
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
    vocab.token_ids.emplace(vocab.dict.at(i), static_cast<int>(i));
  }
}

std::vector<int> Encode(const Vocab& vocab, std::string_view text,
                        bool add_bos, bool add_eos) {
  std::vector<int> tokens;
  tokens.reserve(text.size() + 3);

  if (add_bos) {
    tokens.push_back(1);
  }
  if (!text.empty()) {
    const auto prefix = vocab.token_ids.find(" ");
    if (prefix == vocab.token_ids.end()) {
      throw std::runtime_error("tokenizer is missing the dummy prefix token");
    }
    tokens.push_back(prefix->second);
  }

  std::string codepoint;
  codepoint.reserve(4);
  for (std::size_t i = 0; i < text.size(); ++i) {
    const auto byte = static_cast<unsigned char>(text[i]);
    if ((byte & 0xc0U) != 0x80U) {
      codepoint.clear();
    }
    codepoint.push_back(text[i]);

    const bool next_is_continuation =
        i + 1 < text.size() &&
        (static_cast<unsigned char>(text[i + 1]) & 0xc0U) == 0x80U;
    if (next_is_continuation && codepoint.size() < 4) {
      continue;
    }

    const auto token = vocab.token_ids.find(codepoint);
    if (token != vocab.token_ids.end()) {
      tokens.push_back(token->second);
    } else {
      for (const char value : codepoint) {
        tokens.push_back(static_cast<unsigned char>(value) + 3);
      }
    }
    codepoint.clear();
  }

  while (tokens.size() > 1) {
    float best_score = -std::numeric_limits<float>::infinity();
    int best_token = -1;
    std::size_t best_index = 0;

    for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
      const std::string merged =
          vocab.dict.at(tokens[i]) + vocab.dict.at(tokens[i + 1]);
      const auto token = vocab.token_ids.find(merged);
      if (token != vocab.token_ids.end() &&
          vocab.scores.at(token->second) > best_score) {
        best_score = vocab.scores.at(token->second);
        best_token = token->second;
        best_index = i;
      }
    }

    if (best_token < 0) {
      break;
    }
    tokens.at(best_index) = best_token;
    tokens.erase(tokens.begin() + best_index + 1);
  }

  if (add_eos) {
    tokens.push_back(2);
  }
  return tokens;
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

} // namespace llama2
