//
// Copyright RIME Developers
// Distributed under the BSD License
//

#include "scorer.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>

#ifdef RIME_PERPLEXITY_ENABLE_LLAMA
#include <ggml-backend.h>
#include <ggml-cpu.h>
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

struct TokenizedInput {
  vector<llama_token> tokens;
  int score_start = 0;
};

static int ScoredTokenCount(size_t token_count, int score_start) {
  const int start = std::max(1, score_start);
  return std::max(0, static_cast<int>(token_count) - start);
}

static int ScoredTokenCount(const TokenizedInput& input) {
  return ScoredTokenCount(input.tokens.size(), input.score_start);
}

static size_t ScorePrefixSumStart(int score_start) {
  return static_cast<size_t>(std::max(0, score_start));
}

static ggml_backend_dev_t LoadBackendsForGpuLayers(int gpu_layers) {
  if (gpu_layers != 0) {
    ggml_backend_load_all();
    return nullptr;
  }

  if (!ggml_backend_reg_by_name("CPU")) {
    ggml_backend_register(ggml_backend_cpu_reg());
  }
  ggml_backend_dev_t cpu_dev =
      ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  if (!cpu_dev) {
    LOG(ERROR) << "perplexity: failed to load llama.cpp CPU backend";
  }
  return cpu_dev;
}

class LlamaCausalScorer : public PerplexityScorer {
 public:
  explicit LlamaCausalScorer(const PerplexityScorerOptions& options)
      : max_parallel_(std::max(1, options.batch_size)),
        prefix_cache_capacity_(std::max(0, options.cache_size)),
        unknown_token_penalty_(std::max(0.0, options.unknown_token_penalty)),
        score_prefix_(options.score_prefix),
        score_suffix_(options.score_suffix) {
    ggml_backend_dev_t cpu_dev =
        LoadBackendsForGpuLayers(options.gpu_layers);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = options.gpu_layers;
    ggml_backend_dev_t cpu_devices[] = {cpu_dev, nullptr};
    if (cpu_dev) {
      model_params.devices = cpu_devices;
    }
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
    n_ctx_ = context_params.n_ctx;
    context_params.n_batch = std::max(64, options.max_length);
    context_params.n_ubatch = context_params.n_batch;
    context_params.n_seq_max = max_parallel_ + prefix_cache_capacity_;
    context_params.kv_unified = true;
    context_params.no_perf = true;

    context_ = llama_init_from_model(model_, context_params);
    if (!context_) {
      LOG(ERROR) << "perplexity: failed to create causal LM context";
      llama_model_free(model_);
      model_ = nullptr;
      return;
    }
    ResetSeqIds();
    LOG(INFO) << "perplexity: loaded causal LM: " << options.model_path
              << ", gpu_layers=" << options.gpu_layers
              << ", batch_size=" << max_parallel_
              << ", cache_size=" << prefix_cache_capacity_
              << (prefix_cache_capacity_ > 0
                      ? " (prefix KV cache enabled, scores not "
                        "bit-identical to uncached baseline)"
                      : " (prefix KV cache disabled)")
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

    vector<TokenizedInput> tokenized(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      const string prefix =
          inputs[i].context.empty() ? score_prefix_ : inputs[i].context;
      auto prefix_tokens = Tokenize(prefix);
      auto tokens = Tokenize(prefix + inputs[i].text + score_suffix_);
      const int score_start = static_cast<int>(prefix_tokens.size());
      if (score_start < static_cast<int>(tokens.size())) {
        tokenized[i] = {std::move(tokens), score_start};
        scores[i].token_count = ScoredTokenCount(tokenized[i]);
      }
    }
    ScoreDirect(tokenized, &scores);
    LOG(INFO) << "perplexity: causal_cache hits=" << last_cache_hits_ << "/"
              << last_cache_lookups_
              << " avg_matched_len=" << last_avg_matched_len_;
    return scores;
  }

 private:
  struct CachedPrefix {
    vector<llama_token> tokens;
    vector<double> prefix_logprob_sums;
    llama_seq_id seq_id = -1;
    int length = 0;
    double logprob_sum = 0.0;
    int token_count = 0;
    int64_t last_used = 0;
  };

  struct PrefixMatch {
    int cache_index = -1;
    int matched_len = 0;
  };

  struct ScorePlan {
    size_t candidate_index = 0;
    const vector<llama_token>* tokens = nullptr;
    llama_seq_id seq_id = -1;
    int cache_index = -1;
    int matched_len = 0;
    int score_start = 0;
    int decode_start = 0;
    double cached_logprob_sum = 0.0;
    int cached_token_count = 0;
    vector<double> cached_prefix_sums;
    double suffix_logprob_sum = 0.0;
    int suffix_token_count = 0;
    bool full_cache_hit = false;
  };

  void ScoreDirect(const vector<TokenizedInput>& tokenized,
                   vector<PerplexityScore>* scores) {
    ResetCacheStats();
    size_t start = 0;
    while (start < tokenized.size()) {
      if (tokenized[start].tokens.empty()) {
        ++start;
        continue;
      }

      int total_tokens = 0;
      size_t end = start;
      const int max_tokens =
          std::max(1, static_cast<int>(llama_n_batch(context_)));
      const size_t max_seqs = static_cast<size_t>(max_parallel_);
      while (end < tokenized.size() && end - start < max_seqs) {
        if (tokenized[end].tokens.empty()) {
          ++end;
          continue;
        }
        const int n = static_cast<int>(tokenized[end].tokens.size());
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
    if (text.empty())
      return {};
    try {
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
    } catch (const std::exception& e) {
      LOG(ERROR) << "perplexity: causal tokenize threw on text=\""
                 << text.substr(0, 64) << "\": " << e.what();
      return {};
    }
  }

  double ScoreTokens(const TokenizedInput& input) {
    if (prefix_cache_capacity_ > 0)
      return ScoreTokensWithCache(input);

    const auto& tokens = input.tokens;

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
    if (input.score_start <= 0 && !tokens.empty())
      sum -= TokenPenalty(tokens.front());
    int count = 0;
    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
      if (static_cast<int>(i + 1) >= input.score_start) {
        sum += LogSoftmaxAt(logits + static_cast<size_t>(count) * n_vocab_,
                            n_vocab_, tokens[i + 1]);
        sum -= TokenPenalty(tokens[i + 1]);
      }
      ++count;
    }
    llama_batch_free(batch);
    const int score_count = ScoredTokenCount(input);
    return score_count > 0 ? sum / score_count
                           : -std::numeric_limits<double>::infinity();
  }

  void ScoreTokenBatch(const vector<TokenizedInput>& tokenized,
                       size_t start,
                       size_t end,
                       vector<PerplexityScore>* scores) {
    if (prefix_cache_capacity_ > 0) {
      ScoreTokenBatchWithCache(tokenized, start, end, scores);
      return;
    }

    llama_memory_clear(llama_get_memory(context_), true);

    int total_tokens = 0;
    for (size_t i = start; i < end; ++i) {
      total_tokens += static_cast<int>(tokenized[i].tokens.size());
    }
    if (total_tokens <= 0)
      return;

    llama_batch batch = llama_batch_init(total_tokens, 0, end - start);
    int token_pos = 0;
    for (size_t i = start; i < end; ++i) {
      const auto& tokens = tokenized[i].tokens;
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
        if (!tokenized[i].tokens.empty())
          (*scores)[i].average_logprob = ScoreTokens(tokenized[i]);
      }
      return;
    }

    const float* logits = llama_get_logits(context_);
    int logits_pos = 0;
    for (size_t i = start; i < end; ++i) {
      const auto& input = tokenized[i];
      const auto& tokens = input.tokens;
      double sum = 0.0;
      if (input.score_start <= 0 && !tokens.empty())
        sum -= TokenPenalty(tokens.front());
      for (size_t j = 0; j + 1 < tokens.size(); ++j) {
        if (static_cast<int>(j + 1) >= input.score_start) {
          sum +=
              LogSoftmaxAt(logits + static_cast<size_t>(logits_pos) * n_vocab_,
                           n_vocab_, tokens[j + 1]);
          sum -= TokenPenalty(tokens[j + 1]);
        }
        ++logits_pos;
      }
      const int score_count = ScoredTokenCount(input);
      (*scores)[i].average_logprob =
          score_count > 0 ? sum / score_count
                          : -std::numeric_limits<double>::infinity();
    }
    llama_batch_free(batch);
  }

  double ScoreTokensWithCache(const TokenizedInput& input) {
    vector<TokenizedInput> single{input};
    vector<PerplexityScore> scores(1);
    ScoreTokenBatchWithCache(single, 0, 1, &scores);
    return scores[0].average_logprob;
  }

  void ScoreTokenBatchWithCache(const vector<TokenizedInput>& tokenized,
                                size_t start,
                                size_t end,
                                vector<PerplexityScore>* scores) {
    vector<ScorePlan> plans;
    plans.reserve(end - start);
    int total_decode_tokens = 0;

    for (size_t i = start; i < end; ++i) {
      const auto& input = tokenized[i];
      const auto& tokens = input.tokens;
      if (tokens.empty())
        continue;
      ScorePlan plan;
      plan.candidate_index = i;
      plan.tokens = &tokens;
      plan.score_start = input.score_start;

      PrefixMatch match = FindLongestPrefix(tokens);
      RecordCacheLookup(match.matched_len);
      if (match.cache_index >= 0) {
        TouchPrefix(match.cache_index);
        plan.cache_index = match.cache_index;
        plan.matched_len = match.matched_len;
        if (plan.matched_len == static_cast<int>(tokens.size())) {
          const auto& cached = prefix_cache_[match.cache_index];
          const size_t score_start =
              ScorePrefixSumStart(plan.score_start);
          const double score_sum =
              PrefixLogprobSum(cached, tokens.size()) -
              PrefixLogprobSum(cached, score_start);
          const int score_count =
              ScoredTokenCount(tokens.size(), plan.score_start);
          (*scores)[i].average_logprob =
              score_count > 0
                  ? score_sum / score_count
                  : -std::numeric_limits<double>::infinity();
          plan.full_cache_hit = true;
          plans.push_back(plan);
          continue;
        }

        const auto& cached = prefix_cache_[match.cache_index];
        plan.decode_start = std::max(0, plan.matched_len - 1);
        plan.cached_logprob_sum =
            PrefixLogprobSum(cached, static_cast<size_t>(plan.matched_len));
        plan.cached_token_count = std::max(0, plan.matched_len - 1);
        plan.cached_prefix_sums = cached.prefix_logprob_sums;
        plan.seq_id = AllocSeqId();
        if (plan.decode_start > 0) {
          llama_memory_seq_cp(llama_get_memory(context_), cached.seq_id,
                              plan.seq_id, 0, plan.decode_start);
        }
      } else {
        plan.decode_start = 0;
        plan.seq_id = AllocSeqId();
      }
      if (plan.seq_id < 0) {
        (*scores)[i].average_logprob =
            ScoreTokensWithoutCache({tokens, plan.score_start});
        continue;
      }

      total_decode_tokens +=
          static_cast<int>(tokens.size()) - plan.decode_start;
      plans.push_back(plan);
    }

    if (total_decode_tokens <= 0)
      return;
    EvictUntilTokenCapacity(total_decode_tokens);

    llama_batch batch = llama_batch_init(total_decode_tokens, 0, 1);
    int token_pos = 0;
    for (const auto& plan : plans) {
      if (plan.full_cache_hit)
        continue;
      const auto& tokens = *plan.tokens;
      for (size_t j = static_cast<size_t>(plan.decode_start);
           j < tokens.size(); ++j) {
        batch.token[token_pos] = tokens[j];
        batch.pos[token_pos] = static_cast<llama_pos>(j);
        batch.n_seq_id[token_pos] = 1;
        batch.seq_id[token_pos][0] = plan.seq_id;
        batch.logits[token_pos] = j + 1 < tokens.size();
        ++token_pos;
      }
    }
    batch.n_tokens = token_pos;

    if (llama_decode(context_, batch) != 0) {
      llama_batch_free(batch);
      for (const auto& plan : plans) {
        if (plan.full_cache_hit)
          continue;
        llama_memory_seq_rm(llama_get_memory(context_), plan.seq_id, -1, -1);
        (*scores)[plan.candidate_index].average_logprob =
            ScoreTokensWithoutCache({*plan.tokens, plan.score_start});
      }
      return;
    }

    const float* logits = llama_get_logits(context_);
    int logits_pos = 0;
    for (auto& plan : plans) {
      if (plan.full_cache_hit)
        continue;
      const auto& tokens = *plan.tokens;
      vector<double> prefix_sums(tokens.size() + 1, 0.0);
      if (!plan.cached_prefix_sums.empty()) {
        const size_t copy_len =
            std::min(prefix_sums.size(), plan.cached_prefix_sums.size());
        for (size_t i = 0; i < copy_len; ++i)
          prefix_sums[i] = plan.cached_prefix_sums[i];
      }
      if (plan.decode_start == 0 && !tokens.empty()) {
        const double first_token_penalty = TokenPenalty(tokens.front());
        plan.suffix_logprob_sum -= first_token_penalty;
        prefix_sums[1] = -first_token_penalty;
      }
      for (size_t j = static_cast<size_t>(plan.decode_start);
           j + 1 < tokens.size(); ++j) {
        const double logprob =
            LogSoftmaxAt(logits + static_cast<size_t>(logits_pos) * n_vocab_,
                         n_vocab_, tokens[j + 1]) -
            TokenPenalty(tokens[j + 1]);
        plan.suffix_logprob_sum += logprob;
        ++plan.suffix_token_count;
        prefix_sums[j + 2] = prefix_sums[j + 1] + logprob;
        ++logits_pos;
      }
      const double total_sum =
          plan.cached_logprob_sum + plan.suffix_logprob_sum;
      const int total_count = plan.cached_token_count + plan.suffix_token_count;
      const size_t score_start =
          ScorePrefixSumStart(plan.score_start);
      const double score_sum =
          prefix_sums[tokens.size()] -
          prefix_sums[score_start];
      const int score_count =
          ScoredTokenCount(tokens.size(), plan.score_start);
      (*scores)[plan.candidate_index].average_logprob =
          score_count > 0 ? score_sum / score_count
                          : -std::numeric_limits<double>::infinity();
      InsertOrReplacePrefix(plan.seq_id, tokens, std::move(prefix_sums),
                            total_sum, total_count);
    }
    llama_batch_free(batch);
  }

  double ScoreTokensWithoutCache(const TokenizedInput& input) {
    const int saved_capacity = prefix_cache_capacity_;
    prefix_cache_capacity_ = 0;
    const double score = ScoreTokens(input);
    llama_memory_clear(llama_get_memory(context_), true);
    prefix_cache_.clear();
    ResetSeqIds();
    prefix_cache_capacity_ = saved_capacity;
    return score;
  }

  PrefixMatch FindLongestPrefix(const vector<llama_token>& tokens) const {
    PrefixMatch best;
    for (size_t i = 0; i < prefix_cache_.size(); ++i) {
      const auto& cached = prefix_cache_[i];
      const size_t limit = std::min(tokens.size(), cached.tokens.size());
      size_t len = 0;
      while (len < limit && tokens[len] == cached.tokens[len])
        ++len;
      if (len > static_cast<size_t>(best.matched_len)) {
        best.cache_index = static_cast<int>(i);
        best.matched_len = static_cast<int>(len);
      }
    }
    if (best.matched_len < 2) {
      best.cache_index = -1;
      best.matched_len = 0;
    }
    return best;
  }

  void TouchPrefix(int cache_index) {
    if (cache_index < 0 ||
        static_cast<size_t>(cache_index) >= prefix_cache_.size()) {
      return;
    }
    prefix_cache_[cache_index].last_used = ++access_counter_;
  }

  void EvictLRUIfNeeded() {
    if (prefix_cache_capacity_ <= 0 ||
        prefix_cache_.size() < static_cast<size_t>(prefix_cache_capacity_)) {
      return;
    }
    auto victim = std::min_element(
        prefix_cache_.begin(), prefix_cache_.end(),
        [](const CachedPrefix& a, const CachedPrefix& b) {
          return a.last_used < b.last_used;
        });
    Evict(victim);
  }

  void EvictUntilTokenCapacity(int decode_tokens) {
    while (!prefix_cache_.empty() &&
           CachedTokenCount() + decode_tokens > n_ctx_) {
      auto victim = std::min_element(
          prefix_cache_.begin(), prefix_cache_.end(),
          [](const CachedPrefix& a, const CachedPrefix& b) {
            return a.last_used < b.last_used;
          });
      Evict(victim);
    }
  }

  int CachedTokenCount() const {
    int total = 0;
    for (const auto& cached : prefix_cache_)
      total += cached.length;
    return total;
  }

  void Evict(vector<CachedPrefix>::iterator victim) {
    llama_memory_seq_rm(llama_get_memory(context_), victim->seq_id, -1, -1);
    free_seq_ids_.push_back(victim->seq_id);
    prefix_cache_.erase(victim);
  }

  llama_seq_id AllocSeqId() {
    if (free_seq_ids_.empty())
      EvictLRUIfNeeded();
    if (free_seq_ids_.empty()) {
      llama_memory_clear(llama_get_memory(context_), true);
      prefix_cache_.clear();
      ResetSeqIds();
    }
    if (free_seq_ids_.empty())
      return -1;
    llama_seq_id seq_id = free_seq_ids_.back();
    free_seq_ids_.pop_back();
    return seq_id;
  }

  void ResetSeqIds() {
    free_seq_ids_.clear();
    const int n_seq = max_parallel_ + prefix_cache_capacity_;
    free_seq_ids_.reserve(static_cast<size_t>(n_seq));
    for (int i = n_seq - 1; i >= 0; --i)
      free_seq_ids_.push_back(static_cast<llama_seq_id>(i));
  }

  void InsertOrReplacePrefix(llama_seq_id seq_id,
                             const vector<llama_token>& tokens,
                             vector<double>&& prefix_logprob_sums,
                             double logprob_sum,
                             int token_count) {
    for (auto& cached : prefix_cache_) {
      if (cached.tokens == tokens) {
        if (cached.seq_id != seq_id) {
          llama_memory_seq_rm(llama_get_memory(context_), cached.seq_id, -1,
                              -1);
          free_seq_ids_.push_back(cached.seq_id);
          cached.seq_id = seq_id;
        }
        cached.prefix_logprob_sums = std::move(prefix_logprob_sums);
        cached.length = static_cast<int>(tokens.size());
        cached.logprob_sum = logprob_sum;
        cached.token_count = token_count;
        cached.last_used = ++access_counter_;
        return;
      }
    }

    EvictLRUIfNeeded();
    CachedPrefix cached;
    cached.tokens = tokens;
    cached.prefix_logprob_sums = std::move(prefix_logprob_sums);
    cached.seq_id = seq_id;
    cached.length = static_cast<int>(tokens.size());
    cached.logprob_sum = logprob_sum;
    cached.token_count = token_count;
    cached.last_used = ++access_counter_;
    prefix_cache_.push_back(std::move(cached));
    EvictUntilTokenCapacity(0);
  }

  static double PrefixLogprobSum(const CachedPrefix& cached, size_t length) {
    if (length < cached.prefix_logprob_sums.size())
      return cached.prefix_logprob_sums[length];
    return cached.logprob_sum;
  }

  void ResetCacheStats() {
    last_cache_lookups_ = 0;
    last_cache_hits_ = 0;
    last_cache_matched_total_ = 0;
    last_avg_matched_len_ = 0.0;
  }

  void RecordCacheLookup(int matched_len) {
    ++last_cache_lookups_;
    if (matched_len > 0) {
      ++last_cache_hits_;
      last_cache_matched_total_ += matched_len;
    }
    last_avg_matched_len_ =
        last_cache_lookups_ > 0
            ? static_cast<double>(last_cache_matched_total_) /
                  static_cast<double>(last_cache_lookups_)
            : 0.0;
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
  int n_ctx_ = 1024;
  int max_parallel_ = 1;
  int prefix_cache_capacity_ = 0;
  vector<CachedPrefix> prefix_cache_;
  vector<llama_seq_id> free_seq_ids_;
  int64_t access_counter_ = 0;
  int last_cache_lookups_ = 0;
  int last_cache_hits_ = 0;
  int last_cache_matched_total_ = 0;
  double last_avg_matched_len_ = 0.0;
  double unknown_token_penalty_ = 0.0;
  string score_prefix_;
  string score_suffix_;
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
