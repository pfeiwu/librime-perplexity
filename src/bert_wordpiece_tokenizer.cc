//
// Copyright RIME Developers
// Distributed under the BSD License
//

#include "bert_wordpiece_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace rime {

namespace {

static string LowerAscii(string text) {
  for (char& ch : text) {
    unsigned char c = static_cast<unsigned char>(ch);
    if (c < 128)
      ch = static_cast<char>(std::tolower(c));
  }
  return text;
}

static bool IsAsciiAlnum(const string& ch) {
  return ch.size() == 1 &&
         std::isalnum(static_cast<unsigned char>(ch[0]));
}

static bool IsAsciiWhitespace(const string& ch) {
  return ch.size() == 1 &&
         std::isspace(static_cast<unsigned char>(ch[0]));
}

static vector<string> SplitUtf8(const string& text) {
  vector<string> chars;
  for (size_t i = 0; i < text.size();) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
    size_t len = 1;
    if ((c & 0xe0) == 0xc0) {
      len = 2;
    } else if ((c & 0xf0) == 0xe0) {
      len = 3;
    } else if ((c & 0xf8) == 0xf0) {
      len = 4;
    }
    if (i + len > text.size())
      len = 1;
    chars.push_back(text.substr(i, len));
    i += len;
  }
  return chars;
}

static string DirName(const string& path) {
  const size_t pos = path.find_last_of("/\\");
  return pos == string::npos ? string() : path.substr(0, pos);
}

static string JoinPath(const string& dir, const string& file) {
  if (dir.empty())
    return file;
  const char last = dir[dir.size() - 1];
  if (last == '/' || last == '\\')
    return dir + file;
  return dir + "/" + file;
}

}  // namespace

bool BertWordPieceTokenizer::Load(const string& vocab_path) {
  std::ifstream in(vocab_path);
  if (!in) {
    LOG(ERROR) << "perplexity: failed to open BERT vocab: " << vocab_path;
    return false;
  }
  string token;
  int64_t id = 0;
  while (std::getline(in, token)) {
    if (!token.empty() && token[token.size() - 1] == '\r')
      token.pop_back();
    if (!token.empty())
      vocab_.emplace(token, id);
    ++id;
  }
  cls_id_ = Id("[CLS]");
  sep_id_ = Id("[SEP]");
  mask_id_ = Id("[MASK]");
  unk_id_ = Id("[UNK]");
  pad_id_ = Id("[PAD]");
  LoadTokenizerConfig(vocab_path);
  return cls_id_ >= 0 && sep_id_ >= 0 && mask_id_ >= 0 && unk_id_ >= 0 &&
         pad_id_ >= 0;
}

BertTokenizedText BertWordPieceTokenizer::Tokenize(const string& text,
                                                   int max_tokens) const {
  BertTokenizedText result;
  vector<string> chars = SplitUtf8(text);
  for (size_t i = 0; i < chars.size();) {
    if (IsAsciiWhitespace(chars[i])) {
      ++i;
      continue;
    }
    if (IsAsciiAlnum(chars[i])) {
      string word;
      while (i < chars.size() && IsAsciiAlnum(chars[i])) {
        word += chars[i];
        ++i;
      }
      AddWordPiece(do_lower_case_ ? LowerAscii(word) : word, &result);
    } else {
      const int64_t id = Id(chars[i]);
      if (id >= 0) {
        result.tokens.push_back(id);
      } else {
        result.tokens.push_back(unk_id_);
        ++result.unk_count;
      }
      ++i;
    }
    if (max_tokens > 0 &&
        static_cast<int>(result.tokens.size()) >= max_tokens) {
      result.tokens.resize(max_tokens);
      break;
    }
  }
  return result;
}

int64_t BertWordPieceTokenizer::Id(const string& token) const {
  auto it = vocab_.find(token);
  return it == vocab_.end() ? -1 : it->second;
}

void BertWordPieceTokenizer::AddWordPiece(const string& word,
                                          BertTokenizedText* result) const {
  size_t start = 0;
  vector<int64_t> pieces;
  while (start < word.size()) {
    size_t end = word.size();
    int64_t matched = -1;
    size_t matched_end = start;
    while (end > start) {
      string piece = word.substr(start, end - start);
      if (start > 0)
        piece = "##" + piece;
      matched = Id(piece);
      if (matched >= 0) {
        matched_end = end;
        break;
      }
      --end;
    }
    if (matched < 0) {
      result->tokens.push_back(unk_id_);
      ++result->unk_count;
      return;
    }
    pieces.push_back(matched);
    start = matched_end;
  }
  result->tokens.insert(result->tokens.end(), pieces.begin(), pieces.end());
}

void BertWordPieceTokenizer::LoadTokenizerConfig(const string& vocab_path) {
  do_lower_case_ = false;
  std::ifstream in(JoinPath(DirName(vocab_path), "tokenizer_config.json"));
  if (!in)
    return;
  std::stringstream buffer;
  buffer << in.rdbuf();
  const string config = buffer.str();
  const size_t key = config.find("\"do_lower_case\"");
  if (key == string::npos)
    return;
  const size_t value = config.find("true", key);
  const size_t false_value = config.find("false", key);
  do_lower_case_ = value != string::npos &&
                   (false_value == string::npos || value < false_value);
}

}  // namespace rime
