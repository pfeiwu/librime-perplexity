// SPDX-License-Identifier: BSD-3-Clause

#ifndef RIME_PERPLEXITY_RANKER_H_
#define RIME_PERPLEXITY_RANKER_H_

#include <memory>
#include <rime/filter.h>
#include <rime/gear/filter_commons.h>
#include <rime/translation.h>

#include "scorer.h"

namespace rime {

class PerplexityRanker : public Filter, public TagMatching {
 public:
  explicit PerplexityRanker(const Ticket& ticket);
  ~PerplexityRanker() override;

  an<Translation> Apply(an<Translation> translation,
                        CandidateList* candidates) override;
  bool AppliesToSegment(Segment* segment) override;

 private:
  bool IsRankableCandidate(const an<Candidate>& cand) const;
  size_t CurrentInputSize() const;
  string BuildHistoryContext() const;
  string ConvertForScoring(const string& text) const;
  vector<string> ConvertUnitsForScoring(const vector<string>& units) const;

  int top_k_ = 0;
  int scan_size_ = 50;
  int rank_size_ = 20;
  int min_input_size_ = 0;
  int history_context_commits_ = 0;
  double score_weight_ = 1.0;
  hash_set<string> rank_types_;
  std::unique_ptr<PerplexityScorer> scorer_;
  std::unique_ptr<class ScoreTextConverter> score_text_converter_;
};

}  // namespace rime

#endif  // RIME_PERPLEXITY_RANKER_H_
