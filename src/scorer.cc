//
// Copyright RIME Developers
// Distributed under the BSD License
//

#include "scorer.h"

#include <rime/common.h>

namespace rime {

namespace {

class NullPerplexityScorer : public PerplexityScorer {
 public:
  bool Ready() const override { return false; }

  vector<PerplexityScore> Score(
      const vector<PerplexityInput>& inputs) override {
    return vector<PerplexityScore>(inputs.size());
  }
};

}  // namespace

std::unique_ptr<PerplexityScorer> CreateLlamaCausalScorer(
    const PerplexityScorerOptions& options);
std::unique_ptr<PerplexityScorer> CreateOnnxMaskedScorer(
    const PerplexityScorerOptions& options);

std::unique_ptr<PerplexityScorer> CreatePerplexityScorer(
    const PerplexityScorerOptions& options) {
  if (options.model_path.empty()) {
    return std::make_unique<NullPerplexityScorer>();
  }
  if (options.model_type == PerplexityModelType::kMaskedLm) {
    return CreateOnnxMaskedScorer(options);
  }
  return CreateLlamaCausalScorer(options);
}

}  // namespace rime
