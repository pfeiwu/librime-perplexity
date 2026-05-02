//
// Copyright RIME Developers
// Distributed under the BSD License
//

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

  an<Translation> Apply(an<Translation> translation,
                        CandidateList* candidates) override;
  bool AppliesToSegment(Segment* segment) override;

 private:
  bool IsRankableCandidate(const an<Candidate>& cand) const;
  size_t CurrentInputSize() const;
  string BuildHistoryContext() const;

  int top_k_ = 0;
  int scan_size_ = 50;
  int rank_size_ = 20;
  int min_input_size_ = 0;
  int history_context_commits_ = 0;
  double score_weight_ = 1.0;
  hash_set<string> rank_types_;
  std::unique_ptr<PerplexityScorer> scorer_;
};

}  // namespace rime

#endif  // RIME_PERPLEXITY_RANKER_H_
