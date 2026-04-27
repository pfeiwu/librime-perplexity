//
// Copyright RIME Developers
// Distributed under the BSD License
//

#include <rime/component.h>
#include <rime/registry.h>
#include <rime_api.h>

#include "perplexity_ranker.h"

using namespace rime;

static void rime_perplexity_initialize() {
  Registry& r = Registry::instance();
  r.Register("perplexity_ranker", new Component<PerplexityRanker>);
}

static void rime_perplexity_finalize() {}

RIME_REGISTER_MODULE(perplexity)
