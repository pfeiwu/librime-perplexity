// SPDX-License-Identifier: BSD-3-Clause

#ifndef RIME_PERPLEXITY_BERT_WORDPIECE_TOKENIZER_H_
#define RIME_PERPLEXITY_BERT_WORDPIECE_TOKENIZER_H_

#include <cstdint>
#include <unordered_map>

#include <rime/common.h>

namespace rime {

struct BertTokenizedText {
  vector<int64_t> tokens;
  int unk_count = 0;
};

class BertWordPieceTokenizer {
 public:
  bool Load(const string& vocab_path);

  BertTokenizedText Tokenize(const string& text, int max_tokens = 0) const;

  int64_t cls_id() const { return cls_id_; }
  int64_t sep_id() const { return sep_id_; }
  int64_t mask_id() const { return mask_id_; }
  int64_t unk_id() const { return unk_id_; }
  int64_t pad_id() const { return pad_id_; }
  int64_t vocab_size() const { return static_cast<int64_t>(vocab_.size()); }

 private:
  int64_t Id(const string& token) const;
  void AddWordPiece(const string& word, BertTokenizedText* result) const;
  void LoadTokenizerConfig(const string& vocab_path);

  std::unordered_map<string, int64_t> vocab_;
  bool do_lower_case_ = false;
  int64_t cls_id_ = -1;
  int64_t sep_id_ = -1;
  int64_t mask_id_ = -1;
  int64_t unk_id_ = -1;
  int64_t pad_id_ = -1;
};

}  // namespace rime

#endif  // RIME_PERPLEXITY_BERT_WORDPIECE_TOKENIZER_H_
