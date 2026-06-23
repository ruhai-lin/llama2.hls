#ifndef VOCAB_HPP_
#define VOCAB_HPP_

#include <fstream>
#include <array>
#include <string>
#include <vector>

namespace llama2 {

struct Vocab {
  std::vector<std::string> dict;
  std::array<std::string, 256> byte_pieces;
};

void ResizeVocab(Vocab& vocab, int vocab_size);
void LoadVocab(Vocab& vocab, std::ifstream& fs);
std::string DecodePiece(const Vocab& vocab, int prev_token, int token);

} // namespace llama2

#endif // VOCAB_HPP_
