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

constexpr size_t kMaxDrainCandidates = 256;
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
    config->GetInt(name_space_ + "/size", &size_);
    config->GetDouble(name_space_ + "/score_weight", &score_weight_);
    if (auto types = config->GetList(name_space_ + "/candidate_types")) {
      for (auto it = types->begin(); it != types->end(); ++it) {
        if (auto value = As<ConfigValue>(*it)) {
          candidate_types_.insert(value->str());
        }
      }
    }
    scorer_ = CreateScorer(config, name_space_);
  }
  if (candidate_types_.empty())
    candidate_types_.insert("sentence");
}

bool PerplexityRanker::AppliesToSegment(Segment* segment) {
  return TagsMatch(segment);
}

bool PerplexityRanker::IsRankableCandidate(const an<Candidate>& cand) const {
  return cand && candidate_types_.count(cand->type());
}

an<Translation> PerplexityRanker::Apply(an<Translation> translation,
                                        CandidateList* candidates) {
  if (!translation || translation->exhausted())
    return translation;

  vector<an<Candidate>> rankable;
  rankable.reserve(64);
  while (rankable.size() < kMaxDrainCandidates && !translation->exhausted()) {
    auto cand = translation->Peek();
    if (!IsRankableCandidate(cand))
      break;
    rankable.push_back(cand);
    translation->Next();
  }
  if (rankable.empty())
    return translation;

  if (size_ <= 0) {
    return TranslationFromCandidates(rankable, translation);
  }

  const bool should_rank = scorer_ && scorer_->Ready();
  if (!should_rank) {
    return TranslationFromCandidates(rankable, translation);
  }

  vector<PerplexityInput> inputs;
  inputs.reserve(rankable.size());
  for (const auto& cand : rankable) {
    inputs.push_back({cand->text(), GetCandidateUnits(cand)});
  }

  const auto start = std::chrono::steady_clock::now();
  vector<PerplexityScore> lm_scores = scorer_->Score(inputs);
  const auto finish = std::chrono::steady_clock::now();
  const auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(finish - start)
          .count();
  LOG(INFO) << "perplexity: candidates=" << rankable.size()
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
      return lm;
    return lm * score_weight_ + base_scores[index];
  };

  vector<size_t> order(rankable.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    const double score_a = combined_score(a);
    const double score_b = combined_score(b);
    if (score_a != score_b)
      return score_a > score_b;
    return base_scores[a] > base_scores[b];
  });

  vector<an<Candidate>> reranked;
  reranked.reserve(static_cast<size_t>(size_));
  const size_t promote_count =
      std::min(order.size(), static_cast<size_t>(size_));
  for (size_t i = 0; i < promote_count; ++i) {
    const size_t index = order[i];
    reranked.push_back(rankable[index]);
  }

  return TranslationFromCandidates(reranked, translation);
}

}  // namespace rime
