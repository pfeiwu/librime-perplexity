// SPDX-License-Identifier: BSD-3-Clause

#include "scorer.h"

#include <chrono>
#include <cstdlib>
#include <iostream>

namespace rime {
std::unique_ptr<PerplexityScorer> CreateOnnxMaskedScorer(
    const PerplexityScorerOptions& options);
}  // namespace rime

using rime::PerplexityInput;
using rime::PerplexityScorerOptions;
using rime::string;
using rime::vector;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0]
              << " <model.onnx> [--cache-size N] [--repeat N] <text>...\n";
    return 2;
  }

  PerplexityScorerOptions options;
  options.model_path = argv[1];
  options.batch_size = 32;
  options.max_length = 128;
  options.gpu_layers = 0;
  int repeat = 1;

  int argi = 2;
  while (argi + 1 < argc && argv[argi][0] == '-') {
    string opt = argv[argi++];
    if (opt == "--cache-size") {
      options.cache_size = std::atoi(argv[argi++]);
    } else if (opt == "--repeat") {
      repeat = std::max(1, std::atoi(argv[argi++]));
    } else if (opt == "--gpu-layers") {
      options.gpu_layers = std::atoi(argv[argi++]);
    } else {
      std::cerr << "unknown option: " << opt << "\n";
      return 2;
    }
  }
  if (argi >= argc) {
    std::cerr << "missing text\n";
    return 2;
  }
  vector<PerplexityInput> inputs;
  inputs.reserve(argc - argi);
  for (int i = argi; i < argc; ++i) {
    inputs.push_back({argv[i], {}});
  }

  auto scorer = rime::CreateOnnxMaskedScorer(options);
  if (!scorer || !scorer->Ready()) {
    std::cerr << "masked scorer is not ready\n";
    return 1;
  }

  for (int r = 0; r < repeat; ++r) {
    const auto start = std::chrono::steady_clock::now();
    auto scores = scorer->Score(inputs);
    const auto finish = std::chrono::steady_clock::now();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(finish - start)
            .count();
    std::cout << "run=" << (r + 1) << "\telapsed_ms=" << elapsed_ms << "\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
      std::cout << inputs[i].text << "\t";
      if (i < scores.size()) {
        std::cout << scores[i].average_logprob << "\t"
                  << scores[i].token_count;
      } else {
        std::cout << "nan\t0";
      }
      std::cout << "\n";
    }
  }
  return 0;
}
