//
// Copyright RIME Developers
// Distributed under the BSD License
//

#include "scorer.h"

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef RIME_PERPLEXITY_ENABLE_LLAMA
#include <llama.h>
#endif

namespace rime {

namespace {

static double LogSoftmaxAt(const float* logits,
                           size_t n_vocab,
                           uint32_t token) {
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

#ifdef RIME_PERPLEXITY_ENABLE_LLAMA

class LlamaCausalScorer : public PerplexityScorer {
 public:
  explicit LlamaCausalScorer(const PerplexityScorerOptions& options)
      : unknown_token_penalty_(std::max(0.0, options.unknown_token_penalty)) {
    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = options.gpu_layers;
    model_ =
        llama_model_load_from_file(options.model_path.c_str(), model_params);
    if (!model_) {
      LOG(ERROR) << "perplexity: failed to load causal LM: "
                 << options.model_path;
      return;
    }

    vocab_ = llama_model_get_vocab(model_);
    n_vocab_ = llama_vocab_n_tokens(vocab_);

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max(256, options.max_length);
    context_params.n_batch = std::max(64, options.max_length);
    context_params.n_ubatch = context_params.n_batch;
    context_params.n_seq_max = std::max(1, options.batch_size);
    context_params.kv_unified = true;
    context_params.no_perf = true;

    context_ = llama_init_from_model(model_, context_params);
    if (!context_) {
      LOG(ERROR) << "perplexity: failed to create causal LM context";
      llama_model_free(model_);
      model_ = nullptr;
      return;
    }
    LOG(INFO) << "perplexity: loaded causal LM: " << options.model_path
              << ", gpu_layers=" << options.gpu_layers
              << ", batch_size=" << options.batch_size
              << ", max_length=" << options.max_length;
  }

  ~LlamaCausalScorer() override {
    if (context_)
      llama_free(context_);
    if (model_)
      llama_model_free(model_);
  }

  bool Ready() const override { return model_ && context_ && vocab_; }

  vector<PerplexityScore> Score(
      const vector<PerplexityInput>& inputs) override {
    vector<PerplexityScore> scores(inputs.size());
    if (!Ready())
      return scores;

    vector<vector<llama_token>> tokenized(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto tokens = Tokenize(inputs[i].text);
      if (tokens.size() >= 2) {
        scores[i].token_count = static_cast<int>(tokens.size()) - 1;
        tokenized[i] = std::move(tokens);
      }
    }
    ScoreDirect(tokenized, &scores);
    return scores;
  }

 private:
  void ScoreDirect(const vector<vector<llama_token>>& tokenized,
                   vector<PerplexityScore>* scores) {
    size_t start = 0;
    while (start < tokenized.size()) {
      if (tokenized[start].empty()) {
        ++start;
        continue;
      }

      int total_tokens = 0;
      size_t end = start;
      const int max_tokens =
          std::max(1, static_cast<int>(llama_n_batch(context_)));
      const size_t max_seqs = std::max<uint32_t>(1, llama_n_seq_max(context_));
      while (end < tokenized.size() && end - start < max_seqs) {
        if (tokenized[end].empty()) {
          ++end;
          continue;
        }
        const int n = static_cast<int>(tokenized[end].size());
        if (total_tokens > 0 && total_tokens + n > max_tokens)
          break;
        total_tokens += n;
        ++end;
      }

      if (end == start) {
        (*scores)[start].average_logprob = ScoreTokens(tokenized[start]);
        ++start;
        continue;
      }

      ScoreTokenBatch(tokenized, start, end, scores);
      start = end;
    }
  }

  vector<llama_token> Tokenize(const string& text) const {
    int n = -llama_tokenize(vocab_, text.c_str(), text.size(), nullptr, 0,
                            false, true);
    if (n <= 0)
      return {};
    vector<llama_token> tokens(n);
    if (llama_tokenize(vocab_, text.c_str(), text.size(), tokens.data(),
                       tokens.size(), false, true) < 0) {
      return {};
    }
    return tokens;
  }

  double ScoreTokens(const vector<llama_token>& tokens) {
    llama_memory_clear(llama_get_memory(context_), true);

    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
      batch.token[i] = tokens[i];
      batch.pos[i] = static_cast<llama_pos>(i);
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = i + 1 < tokens.size();
    }
    batch.n_tokens = static_cast<int32_t>(tokens.size());

    if (llama_decode(context_, batch) != 0) {
      llama_batch_free(batch);
      return -std::numeric_limits<double>::infinity();
    }

    const float* logits = llama_get_logits(context_);
    double sum = 0.0;
    int count = 0;
    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
      sum += LogSoftmaxAt(logits + static_cast<size_t>(count) * n_vocab_,
                          n_vocab_, tokens[i + 1]);
      sum -= TokenPenalty(tokens[i + 1]);
      ++count;
    }
    llama_batch_free(batch);
    return count > 0 ? sum / count : -std::numeric_limits<double>::infinity();
  }

  void ScoreTokenBatch(const vector<vector<llama_token>>& tokenized,
                       size_t start,
                       size_t end,
                       vector<PerplexityScore>* scores) {
    llama_memory_clear(llama_get_memory(context_), true);

    int total_tokens = 0;
    for (size_t i = start; i < end; ++i) {
      total_tokens += static_cast<int>(tokenized[i].size());
    }
    if (total_tokens <= 0)
      return;

    llama_batch batch = llama_batch_init(total_tokens, 0, end - start);
    int token_pos = 0;
    for (size_t i = start; i < end; ++i) {
      const auto& tokens = tokenized[i];
      const llama_seq_id seq_id = static_cast<llama_seq_id>(i - start);
      for (size_t j = 0; j < tokens.size(); ++j) {
        batch.token[token_pos] = tokens[j];
        batch.pos[token_pos] = static_cast<llama_pos>(j);
        batch.n_seq_id[token_pos] = 1;
        batch.seq_id[token_pos][0] = seq_id;
        batch.logits[token_pos] = j + 1 < tokens.size();
        ++token_pos;
      }
    }
    batch.n_tokens = token_pos;

    if (llama_decode(context_, batch) != 0) {
      llama_batch_free(batch);
      for (size_t i = start; i < end; ++i) {
        if (!tokenized[i].empty())
          (*scores)[i].average_logprob = ScoreTokens(tokenized[i]);
      }
      return;
    }

    const float* logits = llama_get_logits(context_);
    int logits_pos = 0;
    for (size_t i = start; i < end; ++i) {
      const auto& tokens = tokenized[i];
      double sum = 0.0;
      int count = 0;
      for (size_t j = 0; j + 1 < tokens.size(); ++j) {
        sum += LogSoftmaxAt(logits + static_cast<size_t>(logits_pos) * n_vocab_,
                            n_vocab_, tokens[j + 1]);
        sum -= TokenPenalty(tokens[j + 1]);
        ++logits_pos;
        ++count;
      }
      (*scores)[i].average_logprob =
          count > 0 ? sum / count : -std::numeric_limits<double>::infinity();
    }
    llama_batch_free(batch);
  }

  bool IsUnknown(llama_token token) const {
    return (llama_vocab_get_attr(vocab_, token) & LLAMA_TOKEN_ATTR_UNKNOWN) !=
           0;
  }

  double TokenPenalty(llama_token token) const {
    double penalty = 0.0;
    if (IsUnknown(token))
      penalty += unknown_token_penalty_;
    if (IsByteFallbackToken(token))
      penalty += unknown_token_penalty_;
    return penalty;
  }

  bool IsByteFallbackToken(llama_token token) const {
    int n = llama_token_to_piece(vocab_, token, nullptr, 0, 0, false);
    if (n == 0)
      return false;
    if (n < 0)
      n = -n;
    string piece(static_cast<size_t>(n), '\0');
    n = llama_token_to_piece(vocab_, token, piece.data(), piece.size(), 0,
                             false);
    if (n < 0)
      return true;
    piece.resize(static_cast<size_t>(n));
    return !IsValidUtf8(piece);
  }

  static bool IsValidUtf8(const string& text) {
    const auto* s = reinterpret_cast<const unsigned char*>(text.data());
    const size_t n = text.size();
    for (size_t i = 0; i < n;) {
      const unsigned char c = s[i];
      if (c <= 0x7f) {
        ++i;
      } else if ((c & 0xe0) == 0xc0) {
        if (i + 1 >= n || (s[i + 1] & 0xc0) != 0x80 || c < 0xc2)
          return false;
        i += 2;
      } else if ((c & 0xf0) == 0xe0) {
        if (i + 2 >= n || (s[i + 1] & 0xc0) != 0x80 ||
            (s[i + 2] & 0xc0) != 0x80)
          return false;
        if (c == 0xe0 && s[i + 1] < 0xa0)
          return false;
        if (c == 0xed && s[i + 1] >= 0xa0)
          return false;
        i += 3;
      } else if ((c & 0xf8) == 0xf0) {
        if (i + 3 >= n || (s[i + 1] & 0xc0) != 0x80 ||
            (s[i + 2] & 0xc0) != 0x80 || (s[i + 3] & 0xc0) != 0x80)
          return false;
        if (c == 0xf0 && s[i + 1] < 0x90)
          return false;
        if (c > 0xf4 || (c == 0xf4 && s[i + 1] >= 0x90))
          return false;
        i += 4;
      } else {
        return false;
      }
    }
    return true;
  }

  llama_model* model_ = nullptr;
  llama_context* context_ = nullptr;
  const llama_vocab* vocab_ = nullptr;
  int n_vocab_ = 0;
  double unknown_token_penalty_ = 0.0;
};

#endif  // RIME_PERPLEXITY_ENABLE_LLAMA

}  // namespace

std::unique_ptr<PerplexityScorer> CreateLlamaCausalScorer(
    const PerplexityScorerOptions& options) {
#ifdef RIME_PERPLEXITY_ENABLE_LLAMA
  return std::make_unique<LlamaCausalScorer>(options);
#else
  LOG(WARNING) << "perplexity: model is configured, but the plugin was "
                  "built without llama.cpp support";
  return nullptr;
#endif
}

}  // namespace rime
