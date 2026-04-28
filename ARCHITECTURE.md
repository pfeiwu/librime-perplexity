# rime-perplexity Architecture

`rime-perplexity` is a Rime filter plugin for reranking sentence candidates by
language-model perplexity.  It is intentionally scoped to reranking; sentence
generation remains the job of upstream translators such as `script_translator`.

## Public Component

- Plugin library: `rime-perplexity`
- Module: `perplexity`
- Filter component: `perplexity_ranker`
- Default config namespace: `perplexity`

Typical schema shape:

```yaml
engine:
  translators:
    - script_translator
  filters:
    - uniquifier
    - perplexity_ranker

perplexity:
  candidate_types: [sentence]
  model_type: causal
  model: models/example.gguf
  device: cpu
  max_length: 1024
  batch_size: 32
  cache_size: 0
  score_weight: 60.0
  size: 2
  unknown_token_penalty: 0.0
```

If librime gives the filter the generic namespace `filter` or the component
name `perplexity_ranker`, the plugin reads `perplexity/...` instead.

`model` follows normal Rime resource lookup.  Absolute paths work, while
relative paths are resolved from the user data directory first and the shared
data directory second.

## Data Flow

```text
script_translator
  -> sentence candidates
  -> PerplexityRanker
       drain rankable candidate prefix
       build PerplexityInput{text, units}
       call PerplexityScorer::Score(batch)
       compute lm_score * score_weight + Phrase::weight()
       emit first size reranked candidates
       append the un-drained upstream tail
  -> menu
```

## Internal Boundaries

- `PerplexityRanker`
  - Rime `Filter` implementation.
  - Drains a bounded prefix of upstream candidates whose type is rankable.
  - Computes base grammar score from `Phrase::weight()`.
  - Calls a scorer once with a batch of candidate texts.
  - Sorts by `lm_average_logprob * score_weight + base_score`.
  - Emits the first `size` reranked candidates and drops the remaining drained
    rankable candidates.

- `PerplexityScorer`
  - Model-family abstraction.
  - Input contains both surface text and phrase units.  Causal LM uses text;
    masked LM can later use units for word-aware pseudo-log-likelihood.

- `LlamaCausalScorer`
  - Current runnable backend.
  - Scores causal LM average log probability with llama.cpp.
  - Uses direct per-call batching plus a private cross-call prefix KV cache.
  - Does not use beam reuse, grammar prefilter, or grouped multi-prefix
    batching.

## Caching

`PerplexityRanker` itself is stateless; the underlying
`LlamaCausalScorer` exposes an optional **persistent prefix KV cache**
controlled by `perplexity/cache_size`.

### Default: disabled (`cache_size: 0`)

All scoring is fresh decode with KV cleared between batches. Scores are
deterministic across runs and reproducible regardless of session history.
Recommended for evaluation and as a sane default.

### Opt-in: prefix KV cache (`cache_size: N`)

When `cache_size > 0`, the scorer keeps the most-recently-used N scored
prefixes resident in llama.cpp's KV pool and reuses their KV (via `seq_cp` +
`kv_unified=true`) when a later candidate shares a token prefix.

Measured benefit on incremental-typing input:

- Average per-keystroke latency: -26%
- p50 latency: -50%
- p95 latency: -19%

### Engineering trade-off

Enabling the cache introduces a small numerical perturbation in scores. The
score function semantics are unchanged: the scorer still computes average
log P(token | context) under the same model, but per-candidate
`average_logprob` shifts by up to ~0.1 versus `cache_size: 0`.

This is the same kind of trade-off as choosing a quantized model (q4_K_M shifts
logprob by 1-3) or switching backends (CPU vs CUDA shifts logprob by ~1e-2).
The cache shift is smaller than quantization and larger than backend change.
Empirically:

- Top-1 ranking impact on a 246-sentence benchmark: <= 0.4%
- The drift is fully deterministic: identical input produces identical
  drifted output across runs.
- Discrimination is preserved; only the absolute score values shift.

If you need bit-identical scoring for evaluation, set `cache_size: 0`. For
interactive IME use, the latency improvement is the right trade-off and the
drift is below user-perceptible thresholds.

### Why the drift exists

llama.cpp's unified KV pool (`kv_unified=true`) does not produce strictly
seq-isolated attention numerics when multiple sequences co-reside in the pool:
even fresh-decode candidates that do not reuse cached KV are affected by what
is resident. The sister mode (`kv_unified=false`) does isolate sequences but
rejects partial `seq_cp`, which our prefix-reuse strategy requires. A
reproducer has been filed upstream as <ISSUE_LINK_PLACEHOLDER>; if upstream
resolves either path, this drift will disappear automatically.

## Parameters

| Key | Default | Meaning |
| --- | --- | --- |
| `perplexity/model_type` | `causal` | Scorer family: `causal` now, `masked` planned. |
| `perplexity/model` | empty | Rime resource path to the model. Empty disables scoring. |
| `perplexity/device` | `cpu` | `cpu` maps to `gpu_layers=0`; `gpu` maps to `gpu_layers=-1`. |
| `perplexity/gpu_layers` | unset | Advanced override for llama.cpp GPU offload. |
| `perplexity/max_length` | `1024` | Maximum model context length. |
| `perplexity/batch_size` | `32` | Maximum candidate sequences scored in one batch. |
| `perplexity/cache_size` | `0` | Prefix KV cache capacity. `0` disables caching. |
| `perplexity/score_weight` | `1.0` | Multiplier for LM average log probability before adding base score. |
| `perplexity/unknown_token_penalty` | `0.0` | Penalty for unknown or byte-fallback tokens. |
| `perplexity/candidate_types` | `[sentence]` | Candidate types this filter reranks. |
| `perplexity/size` | `2` | Number of reranked candidates promoted to the front. |

## Planned Scorer Families

- `causal`: left-to-right LM perplexity.
- `masked`: masked LM pseudo-log-likelihood.  The interface is already shaped
  for this, but the C++ backend is not implemented yet.
- Encoder-decoder LM is out of scope for now.
