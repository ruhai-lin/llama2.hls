#ifndef VOCAB_HPP_
#define VOCAB_HPP_

#include <fstream>
#include <array>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace llama2 {

struct Vocab {
  std::vector<std::string> dict;
  std::vector<float> scores;
  std::unordered_map<std::string, int> token_ids;
  std::array<std::string, 256> byte_pieces;
};

void ResizeVocab(Vocab& vocab, int vocab_size);
void LoadVocab(Vocab& vocab, std::ifstream& fs);
std::vector<int> Encode(const Vocab& vocab, std::string_view text,
                        bool add_bos, bool add_eos);
std::string DecodePiece(const Vocab& vocab, int prev_token, int token);

} // namespace llama2

#endif // VOCAB_HPP_
