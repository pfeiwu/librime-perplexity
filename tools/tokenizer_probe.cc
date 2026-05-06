// SPDX-License-Identifier: BSD-3-Clause

#include "bert_wordpiece_tokenizer.h"

#include <iostream>

using rime::BertWordPieceTokenizer;
using rime::string;
using rime::vector;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <vocab.txt> <text>...\n";
    return 2;
  }

  BertWordPieceTokenizer tokenizer;
  if (!tokenizer.Load(argv[1]))
    return 1;

  for (int i = 2; i < argc; ++i) {
    const string text = argv[i];
    auto tokenized = tokenizer.Tokenize(text);
    std::cout << text << "\t";
    std::cout << "[";
    for (size_t j = 0; j < tokenized.tokens.size(); ++j) {
      if (j)
        std::cout << ", ";
      std::cout << tokenized.tokens[j];
    }
    std::cout << "]\n";
  }
  return 0;
}
