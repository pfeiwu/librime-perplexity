//
// Copyright RIME Developers
// Distributed under the BSD License
//

#include "perplexity_ranker.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <rime/candidate.h>
#include <rime/commit_history.h>
#include <rime/config.h>
#include <rime/context.h>
#include <rime/engine.h>
#include <rime/resource.h>
#include <rime/schema.h>
#include <rime/segmentation.h>
#include <rime/service.h>
#include <rime/gear/translator_commons.h>

namespace rime {

namespace {

const ResourceType kModelFileType = {"perplexity_model", "", ""};

static string NormalizeModelType(string value) {
  std::replace(value.begin(), value.end(), '-', '_');
  return value;
}

static PerplexityModelType ParseModelType(const string& raw) {
  string value = NormalizeModelType(raw);
  if (value == "masked" || value == "masked_lm" || value == "mlm")
    return PerplexityModelType::kMaskedLm;
  return PerplexityModelType::kCausalLm;
}

static int GpuLayersForDevice(const string& device) {
  if (device == "gpu")
    return -1;
  if (device != "cpu") {
    LOG(WARNING) << "perplexity: unsupported device: " << device
                 << "; using cpu";
  }
  return 0;
}

static double GetBaseScore(const an<Candidate>& cand) {
  auto genuine = Candidate::GetGenuineCandidate(cand);
  if (auto phrase = As<Phrase>(genuine)) {
    return phrase->weight();
  }
  return cand ? cand->quality() : 0.0;
}

static vector<string> GetCandidateUnits(const an<Candidate>& cand) {
  vector<string> units;
  auto genuine = Candidate::GetGenuineCandidate(cand);
  if (auto sentence = As<Sentence>(genuine)) {
    units.reserve(sentence->components().size());
    for (const auto& component : sentence->components()) {
      units.push_back(component.text);
    }
  } else if (auto phrase = As<Phrase>(genuine)) {
    units.push_back(phrase->text());
  } else if (cand) {
    units.push_back(cand->text());
  }
  if (units.empty() && cand) {
    units.push_back(cand->text());
  }
  return units;
}

static size_t GetCandidateInputSize(const an<Candidate>& cand) {
  if (!cand || cand->end() < cand->start())
    return 0;
  return cand->end() - cand->start();
}

static an<Translation> TranslationFromCandidates(
    const vector<an<Candidate>>& candidates,
    an<Translation> remaining) {
  auto prefix = New<FifoTranslation>();
  for (const auto& cand : candidates) {
    prefix->Append(cand);
  }
  if (remaining && !remaining->exhausted()) {
    return prefix + remaining;
  }
  return prefix;
}

static std::unique_ptr<PerplexityScorer> CreateScorer(Config* config,
                                                      const string& ns) {
  PerplexityScorerOptions options;
  string model_type = "causal";
  config->GetString(ns + "/model_type", &model_type);
  options.model_type = ParseModelType(model_type);
  config->GetString(ns + "/model", &options.model_path);
  if (!options.model_path.empty()) {
    the<ResourceResolver> resolver(
        Service::instance().CreateResourceResolver(kModelFileType));
    options.model_path = resolver->ResolvePath(options.model_path).u8string();
  }
  config->GetInt(ns + "/max_length", &options.max_length);
  config->GetInt(ns + "/batch_size", &options.batch_size);
  config->GetInt(ns + "/cache_size", &options.cache_size);
  string device = "cpu";
  config->GetString(ns + "/device", &device);
  options.gpu_layers = GpuLayersForDevice(device);
  config->GetInt(ns + "/gpu_layers", &options.gpu_layers);
  config->GetDouble(ns + "/unknown_token_penalty",
                    &options.unknown_token_penalty);
  config->GetString(ns + "/score_prefix", &options.score_prefix);
  config->GetString(ns + "/score_suffix", &options.score_suffix);
  return CreatePerplexityScorer(options);
}

}  // namespace

PerplexityRanker::PerplexityRanker(const Ticket& ticket)
    : Filter(ticket), TagMatching(ticket) {
  if (name_space_.empty() || name_space_ == "filter" ||
      name_space_ == "perplexity_ranker") {
    name_space_ = "perplexity";
  }
  if (!engine_)
    return;
  if (Config* config = engine_->schema()->config()) {
    config->GetInt(name_space_ + "/top_k", &top_k_);
    config->GetInt(name_space_ + "/scan_size", &scan_size_);
    config->GetInt(name_space_ + "/rank_size", &rank_size_);
    config->GetInt(name_space_ + "/min_input_size", &min_input_size_);
    config->GetInt(name_space_ + "/history_context_commits",
                   &history_context_commits_);
    config->GetDouble(name_space_ + "/score_weight", &score_weight_);
    if (auto types = config->GetList(name_space_ + "/rank_types")) {
      for (auto it = types->begin(); it != types->end(); ++it) {
        if (auto value = As<ConfigValue>(*it)) {
          rank_types_.insert(value->str());
        }
      }
    }
    scorer_ = CreateScorer(config, name_space_);
  }
  if (rank_types_.empty())
    rank_types_.insert("sentence");
  top_k_ = std::max(0, top_k_);
  scan_size_ = std::max(0, scan_size_);
  rank_size_ = std::max(0, rank_size_);
  min_input_size_ = std::max(0, min_input_size_);
  history_context_commits_ = std::max(0, history_context_commits_);
}

bool PerplexityRanker::AppliesToSegment(Segment* segment) {
  return TagsMatch(segment);
}

bool PerplexityRanker::IsRankableCandidate(const an<Candidate>& cand) const {
  if (!cand)
    return false;
  auto genuine = Candidate::GetGenuineCandidate(cand);
  return genuine && rank_types_.count(genuine->type());
}

size_t PerplexityRanker::CurrentInputSize() const {
  if (!engine_)
    return 0;
  Context* context = engine_->context();
  if (!context)
    return 0;
  const Composition& composition = context->composition();
  if (!composition.empty()) {
    const Segment& segment = composition.back();
    return segment.end >= segment.start ? segment.end - segment.start : 0;
  }
  return context->input().length();
}

string PerplexityRanker::BuildHistoryContext() const {
  if (!engine_ || history_context_commits_ <= 0)
    return string();
  Context* context = engine_->context();
  if (!context)
    return string();
  const CommitHistory& history = context->commit_history();
  if (history.empty())
    return string();

  vector<string> records;
  records.reserve(static_cast<size_t>(history_context_commits_));
  for (auto it = history.rbegin();
       it != history.rend() &&
       records.size() < static_cast<size_t>(history_context_commits_);
       ++it) {
    if (!it->text.empty())
      records.push_back(it->text);
  }

  string result;
  for (auto it = records.rbegin(); it != records.rend(); ++it)
    result += *it;
  return result;
}

an<Translation> PerplexityRanker::Apply(an<Translation> translation,
                                        CandidateList* candidates) {
  if (!translation || translation->exhausted())
    return translation;

  const bool should_rank =
      scorer_ && scorer_->Ready() && scan_size_ > 0 && rank_size_ > 0 &&
      CurrentInputSize() >= static_cast<size_t>(min_input_size_);
  if (!should_rank)
    return translation;

  vector<an<Candidate>> rankable;
  vector<an<Candidate>> scanned;
  rankable.reserve(static_cast<size_t>(rank_size_));
  scanned.reserve(static_cast<size_t>(scan_size_));
  while (scanned.size() < static_cast<size_t>(scan_size_) &&
         rankable.size() < static_cast<size_t>(rank_size_) &&
         !translation->exhausted()) {
    auto cand = translation->Peek();
    scanned.push_back(cand);
    if (IsRankableCandidate(cand))
      rankable.push_back(cand);
    translation->Next();
  }
  if (rankable.empty())
    return TranslationFromCandidates(scanned, translation);

  const string history_context = BuildHistoryContext();
  vector<PerplexityInput> inputs;
  inputs.reserve(rankable.size());
  for (const auto& cand : rankable) {
    inputs.push_back({history_context, cand->text(), GetCandidateUnits(cand)});
  }

  const auto start = std::chrono::steady_clock::now();
  vector<PerplexityScore> lm_scores = scorer_->Score(inputs);
  const auto finish = std::chrono::steady_clock::now();
  const auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(finish - start)
          .count();
  LOG(INFO) << "perplexity: scanned=" << scanned.size()
            << " rankable=" << rankable.size()
            << " input_size=" << CurrentInputSize()
            << " min_input_size=" << min_input_size_
            << " history_context_commits=" << history_context_commits_
            << " history_context_chars=" << history_context.size()
            << " scoring=" << elapsed_ms << "ms";

  vector<double> base_scores(rankable.size(), 0.0);
  for (size_t i = 0; i < rankable.size(); ++i) {
    base_scores[i] = GetBaseScore(rankable[i]);
  }

  auto combined_score = [&](size_t index) {
    double lm = index < lm_scores.size()
                    ? lm_scores[index].average_logprob
                    : -std::numeric_limits<double>::infinity();
    if (!std::isfinite(lm))
      return base_scores[index];
    return lm * score_weight_ + base_scores[index];
  };

  vector<size_t> order(rankable.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    const size_t input_size_a = GetCandidateInputSize(rankable[a]);
    const size_t input_size_b = GetCandidateInputSize(rankable[b]);
    if (input_size_a != input_size_b)
      return input_size_a > input_size_b;
    const double score_a = combined_score(a);
    const double score_b = combined_score(b);
    if (score_a != score_b)
      return score_a > score_b;
    return base_scores[a] > base_scores[b];
  });

  const size_t debug_count = std::min(order.size(), static_cast<size_t>(20));
  for (size_t rank = 0; rank < debug_count; ++rank) {
    const size_t index = order[rank];
    const double lm =
        index < lm_scores.size()
            ? lm_scores[index].average_logprob
            : -std::numeric_limits<double>::infinity();
    auto genuine = Candidate::GetGenuineCandidate(rankable[index]);
    LOG(INFO) << "perplexity_score rank=" << (rank + 1)
              << " src_rank=" << (index + 1)
              << " llm=" << lm
              << " base=" << base_scores[index]
              << " total=" << combined_score(index)
              << " tokens="
              << (index < lm_scores.size() ? lm_scores[index].token_count : 0)
              << " input_size=" << GetCandidateInputSize(rankable[index])
              << " type=" << (genuine ? genuine->type() : string())
              << " outer_type=" << rankable[index]->type()
              << " text=" << rankable[index]->text();
  }

  vector<an<Candidate>> ranked;
  const size_t promote_count =
      top_k_ > 0 ? std::min(order.size(), static_cast<size_t>(top_k_))
                 : order.size();
  ranked.reserve(promote_count);
  for (size_t i = 0; i < promote_count; ++i) {
    const size_t index = order[i];
    ranked.push_back(rankable[index]);
  }

  vector<an<Candidate>> output;
  output.reserve(scanned.size());
  size_t ranked_pos = 0;
  for (const auto& cand : scanned) {
    if (!cand)
      continue;
    if (IsRankableCandidate(cand)) {
      if (ranked_pos < ranked.size())
        output.push_back(ranked[ranked_pos++]);
    } else {
      output.push_back(cand);
    }
  }

  return TranslationFromCandidates(output, translation);
}

}  // namespace rime
