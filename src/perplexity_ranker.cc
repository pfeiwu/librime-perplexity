//
// Copyright RIME Developers
// Distributed under the BSD License
//

#include "perplexity_ranker.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
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
  if (auto sentence = As<Sentence>(genuine)) {
    return sentence->weight();
  }
  return cand ? cand->quality() : 0.0;
}

struct RankScore {
  double base = 0.0;
  double lm = -std::numeric_limits<double>::infinity();
  double base_norm = 0.5;
  double lm_norm = 0.5;
  double adjusted_quality = 0.5;
};

static double NormalizeWithinRange(double value, double min, double max) {
  if (max == min)
    return 0.5;
  return (value - min) / (max - min);
}

static bool SameInputSpan(const an<Candidate>& a, const an<Candidate>& b) {
  if (!a || !b)
    return false;
  return a->start() == b->start() && a->end() == b->end();
}

static void BlendScoresForSpanGroup(const vector<an<Candidate>>& rankable,
                                    const vector<PerplexityScore>& lm_scores,
                                    double score_weight,
                                    size_t start,
                                    vector<bool>* visited,
                                    vector<RankScore>* scores) {
  vector<size_t> group;
  for (size_t i = start; i < rankable.size(); ++i) {
    if ((*visited)[i] || !SameInputSpan(rankable[start], rankable[i]))
      continue;
    (*visited)[i] = true;
    group.push_back(i);
  }

  double base_min = std::numeric_limits<double>::infinity();
  double base_max = -std::numeric_limits<double>::infinity();
  double lm_min = std::numeric_limits<double>::infinity();
  double lm_max = -std::numeric_limits<double>::infinity();
  bool has_lm = false;
  for (size_t index : group) {
    RankScore& score = (*scores)[index];
    score.base = GetBaseScore(rankable[index]);
    score.lm = index < lm_scores.size()
                   ? lm_scores[index].average_logprob
                   : -std::numeric_limits<double>::infinity();
    base_min = std::min(base_min, score.base);
    base_max = std::max(base_max, score.base);
    if (std::isfinite(score.lm)) {
      lm_min = std::min(lm_min, score.lm);
      lm_max = std::max(lm_max, score.lm);
      has_lm = true;
    }
  }

  for (size_t index : group) {
    RankScore& score = (*scores)[index];
    score.base_norm = NormalizeWithinRange(score.base, base_min, base_max);
    score.lm_norm =
        has_lm && std::isfinite(score.lm)
            ? NormalizeWithinRange(score.lm, lm_min, lm_max)
            : score.base_norm;
    score.adjusted_quality =
        (1.0 - score_weight) * score.base_norm + score_weight * score.lm_norm;
  }
}

static vector<RankScore> ComputeAdjustedQuality(
    const vector<an<Candidate>>& rankable,
    const vector<PerplexityScore>& lm_scores,
    double score_weight) {
  vector<RankScore> scores(rankable.size());
  vector<bool> visited(rankable.size(), false);
  for (size_t i = 0; i < rankable.size(); ++i) {
    if (!visited[i]) {
      BlendScoresForSpanGroup(rankable, lm_scores, score_weight, i, &visited,
                              &scores);
    }
  }
  return scores;
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
  score_weight_ = std::max(0.0, std::min(1.0, score_weight_));
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
  if (!engine_)
    return string();
  Context* context = engine_->context();
  if (!context)
    return string();

  vector<string> records;
  if (history_context_commits_ > 0) {
    const CommitHistory& history = context->commit_history();
    records.reserve(static_cast<size_t>(history_context_commits_));
    for (auto it = history.rbegin();
         it != history.rend() &&
         records.size() < static_cast<size_t>(history_context_commits_);
         ++it) {
      if (!it->text.empty())
        records.push_back(it->text);
    }
  }

  string result;
  for (auto it = records.rbegin(); it != records.rend(); ++it)
    result += *it;

  const Composition& composition = context->composition();
  if (composition.empty())
    return result;
  const size_t active_start = composition.back().start;
  for (const Segment& segment : composition) {
    if (segment.end > active_start)
      continue;
    if (segment.status < Segment::kSelected)
      continue;
    if (auto cand = segment.GetSelectedCandidate()) {
      result += cand->text();
    } else if (!segment.HasTag("phony") &&
               segment.end <= composition.input().size()) {
      result +=
          composition.input().substr(segment.start, segment.end - segment.start);
    }
  }
  return result;
}

an<Translation> PerplexityRanker::Apply(an<Translation> translation,
                                        CandidateList* candidates) {
  if (!translation || translation->exhausted())
    return translation;

  const bool input_ready =
      scan_size_ > 0 && CurrentInputSize() >= static_cast<size_t>(min_input_size_);
  if (!input_ready)
    return translation;
  const bool should_score = rank_size_ > 0;
  if (should_score && !(scorer_ && scorer_->Ready()))
    return translation;
  if (!should_score && top_k_ == 0)
    return translation;

  vector<an<Candidate>> rankable;
  vector<an<Candidate>> scanned;
  rankable.reserve(static_cast<size_t>(should_score ? rank_size_ : scan_size_));
  scanned.reserve(static_cast<size_t>(scan_size_));
  while (scanned.size() < static_cast<size_t>(scan_size_) &&
         (!should_score || rankable.size() < static_cast<size_t>(rank_size_)) &&
         !translation->exhausted()) {
    auto cand = translation->Peek();
    scanned.push_back(cand);
    if (IsRankableCandidate(cand))
      rankable.push_back(cand);
    translation->Next();
  }
  if (rankable.empty())
    return TranslationFromCandidates(scanned, translation);

  if (!should_score) {
    vector<an<Candidate>> output;
    output.reserve(scanned.size());
    size_t kept = 0;
    for (const auto& cand : scanned) {
      if (!cand)
        continue;
      if (IsRankableCandidate(cand)) {
        if (kept < static_cast<size_t>(top_k_))
          output.push_back(cand);
        ++kept;
      } else {
        output.push_back(cand);
      }
    }
    return TranslationFromCandidates(output, translation);
  }

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
            << " score_weight=" << score_weight_
            << " history_context_commits=" << history_context_commits_
            << " history_context_chars=" << history_context.size()
            << " scoring=" << elapsed_ms << "ms";

  const vector<RankScore> rank_scores =
      ComputeAdjustedQuality(rankable, lm_scores, score_weight_);

  vector<size_t> order(rankable.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    const size_t len_a = GetCandidateInputSize(rankable[a]);
    const size_t len_b = GetCandidateInputSize(rankable[b]);
    if (len_a != len_b)
      return len_a > len_b;
    return rank_scores[a].adjusted_quality > rank_scores[b].adjusted_quality;
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
              << " lm_norm=" << rank_scores[index].lm_norm
              << " base=" << rank_scores[index].base
              << " base_norm=" << rank_scores[index].base_norm
              << " adjusted_quality=" << rank_scores[index].adjusted_quality
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
