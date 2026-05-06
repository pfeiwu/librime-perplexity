// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <llama.h>

namespace {

double LogSoftmaxAt(const float* logits, size_t n_vocab, llama_token token) {
  float max_logit = logits[0];
  for (size_t i = 1; i < n_vocab; ++i) {
    max_logit = std::max(max_logit, logits[i]);
  }
  double sum = 0.0;
  for (size_t i = 0; i < n_vocab; ++i) {
    sum += std::exp(static_cast<double>(logits[i] - max_logit));
  }
  return static_cast<double>(logits[token] - max_logit) - std::log(sum);
}

std::string Piece(const llama_vocab* vocab, llama_token token) {
  int n = llama_token_to_piece(vocab, token, nullptr, 0, 0, true);
  if (n == 0)
    return {};
  if (n < 0)
    n = -n;
  std::string piece(static_cast<size_t>(n), '\0');
  n = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
  if (n < 0)
    return "<byte-fallback>";
  piece.resize(static_cast<size_t>(n));
  return piece;
}

std::vector<llama_token> Tokenize(const llama_vocab* vocab,
                                  const std::string& text) {
  int n =
      -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, false, true);
  if (n <= 0)
    return {};
  std::vector<llama_token> tokens(static_cast<size_t>(n));
  n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(),
                     tokens.size(), false, true);
  if (n < 0)
    return {};
  tokens.resize(static_cast<size_t>(n));
  return tokens;
}

void Usage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " <model.gguf> [--gpu-layers N] [--max-length N]"
               " [--batch-size N] [--cache-size N]"
               " [--prefix TEXT] [--suffix TEXT]"
               " [--unknown-token-penalty N] <text>...\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    Usage(argv[0]);
    return 2;
  }

  std::string model_path = argv[1];
  int gpu_layers = 0;
  int max_length = 1024;
  int batch_size = 32;
  int cache_size = 0;
  double unknown_token_penalty = 20.0;
  std::string prefix = "，";
  std::string suffix = "，";

  int argi = 2;
  while (argi + 1 < argc && argv[argi][0] == '-') {
    std::string opt = argv[argi++];
    if (opt == "--gpu-layers") {
      gpu_layers = std::atoi(argv[argi++]);
    } else if (opt == "--max-length") {
      max_length = std::atoi(argv[argi++]);
    } else if (opt == "--batch-size") {
      batch_size = std::atoi(argv[argi++]);
    } else if (opt == "--cache-size") {
      cache_size = std::atoi(argv[argi++]);
    } else if (opt == "--prefix") {
      prefix = argv[argi++];
    } else if (opt == "--suffix") {
      suffix = argv[argi++];
    } else if (opt == "--unknown-token-penalty") {
      unknown_token_penalty = std::atof(argv[argi++]);
    } else {
      std::cerr << "unknown option: " << opt << "\n";
      return 2;
    }
  }
  if (argi >= argc) {
    Usage(argv[0]);
    return 2;
  }

  ggml_backend_load_all();

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = gpu_layers;
  llama_model* model =
      llama_model_load_from_file(model_path.c_str(), model_params);
  if (!model) {
    std::cerr << "failed to load model: " << model_path << "\n";
    return 1;
  }

  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = std::max(256, max_length);
  context_params.n_batch = std::max(64, max_length);
  context_params.n_ubatch = context_params.n_batch;
  context_params.n_seq_max = std::max(1, batch_size + cache_size);
  context_params.kv_unified = true;
  context_params.no_perf = true;
  llama_context* context = llama_init_from_model(model, context_params);
  if (!context) {
    std::cerr << "failed to create context\n";
    llama_model_free(model);
    return 1;
  }

  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int n_vocab = llama_vocab_n_tokens(vocab);

  for (int i = argi; i < argc; ++i) {
    const std::string text = argv[i];
    const std::string scored_text = prefix + text + suffix;
    const auto tokens = Tokenize(vocab, scored_text);
    std::cout << "text=" << text << "\n";
    std::cout << "score_prefix=" << prefix << "\n";
    std::cout << "score_suffix=" << suffix << "\n";
    std::cout << "scored_text=" << scored_text << "\n";
    std::cout << "tokens=[";
    for (size_t j = 0; j < tokens.size(); ++j) {
      if (j)
        std::cout << ", ";
      std::cout << tokens[j] << ":" << Piece(vocab, tokens[j]);
    }
    std::cout << "]\n";

    if (tokens.size() < 2) {
      std::cout << "average_logprob=-inf token_count=0\n\n";
      continue;
    }

    llama_memory_clear(llama_get_memory(context), true);
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t j = 0; j < tokens.size(); ++j) {
      batch.token[j] = tokens[j];
      batch.pos[j] = static_cast<llama_pos>(j);
      batch.n_seq_id[j] = 1;
      batch.seq_id[j][0] = 0;
      batch.logits[j] = j + 1 < tokens.size();
    }
    batch.n_tokens = static_cast<int32_t>(tokens.size());

    if (llama_decode(context, batch) != 0) {
      llama_batch_free(batch);
      std::cout << "decode_failed\n\n";
      continue;
    }

    const float* logits = llama_get_logits(context);
    auto token_penalty = [&](llama_token token) {
      double penalty = 0.0;
      if ((llama_vocab_get_attr(vocab, token) & LLAMA_TOKEN_ATTR_UNKNOWN) != 0)
        penalty += unknown_token_penalty;
      if (Piece(vocab, token) == "<byte-fallback>")
        penalty += unknown_token_penalty;
      return penalty;
    };

    double sum = tokens.empty() ? 0.0 : -token_penalty(tokens.front());
    int count = 0;
    std::cout << std::fixed << std::setprecision(6);
    if (!tokens.empty() && token_penalty(tokens.front()) > 0.0) {
      std::cout << "token_penalty index=0 token=" << tokens.front()
                << " piece=" << Piece(vocab, tokens.front())
                << " penalty=" << token_penalty(tokens.front()) << "\n";
    }
    for (size_t j = 0; j + 1 < tokens.size(); ++j) {
      const double logprob =
          LogSoftmaxAt(logits + static_cast<size_t>(count) * n_vocab, n_vocab,
                       tokens[j + 1]);
      const double penalty = token_penalty(tokens[j + 1]);
      sum += logprob - penalty;
      ++count;
      std::cout << "token_score index=" << (j + 1)
                << " token=" << tokens[j + 1]
                << " piece=" << Piece(vocab, tokens[j + 1])
                << " prev=" << tokens[j] << ":" << Piece(vocab, tokens[j])
                << " logprob=" << logprob;
      if (penalty > 0.0)
        std::cout << " penalty=" << penalty;
      std::cout << "\n";
    }
    std::cout << "average_logprob=" << (sum / count)
              << " token_count=" << count << "\n\n";
    llama_batch_free(batch);
  }

  llama_free(context);
  llama_model_free(model);
  return 0;
}
