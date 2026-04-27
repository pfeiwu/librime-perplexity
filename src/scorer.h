//
// Copyright RIME Developers
// Distributed under the BSD License
//

#ifndef RIME_PERPLEXITY_SCORER_H_
#define RIME_PERPLEXITY_SCORER_H_

#include <limits>
#include <memory>
#include <rime/common.h>

namespace rime {

enum class PerplexityModelType {
  kCausalLm,
  kMaskedLm,
};

struct PerplexityInput {
  string text;
  vector<string> units;
};

struct PerplexityScore {
  double average_logprob = -std::numeric_limits<double>::infinity();
  int token_count = 0;

  bool ok() const { return std::isfinite(average_logprob); }
};

struct PerplexityScorerOptions {
  PerplexityModelType model_type = PerplexityModelType::kCausalLm;
  string model_path;
  int max_length = 1024;
  int batch_size = 32;
  int gpu_layers = 0;
  double unknown_token_penalty = 0.0;
};

class PerplexityScorer {
 public:
  virtual ~PerplexityScorer() = default;
  virtual bool Ready() const = 0;
  virtual vector<PerplexityScore> Score(
      const vector<PerplexityInput>& inputs) = 0;
};

std::unique_ptr<PerplexityScorer> CreatePerplexityScorer(
    const PerplexityScorerOptions& options);

}  // namespace rime

#endif  // RIME_PERPLEXITY_SCORER_H_
