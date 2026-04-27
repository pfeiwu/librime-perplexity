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
       promote first size reranked candidates
       append remaining candidates in upstream order
  -> menu
```

## Internal Boundaries

- `PerplexityRanker`
  - Rime `Filter` implementation.
  - Drains a bounded prefix of upstream candidates whose type is rankable.
  - Computes base grammar score from `Phrase::weight()`.
  - Calls a scorer once with a batch of candidate texts.
  - Sorts by `lm_average_logprob * score_weight + base_score`.
  - Promotes the first `size` reranked candidates and keeps the rest in the
    upstream order.

- `PerplexityScorer`
  - Model-family abstraction.
  - Input contains both surface text and phrase units.  Causal LM uses text;
    masked LM can later use units for word-aware pseudo-log-likelihood.

- `LlamaCausalScorer`
  - Current runnable backend.
  - Scores causal LM average log probability with llama.cpp.
  - Uses direct per-call batching only.  No cross-keystroke cache, beam reuse,
    prefix KV sharing, grammar prefilter, or other experimental optimizations.

## Parameters

| Key | Default | Meaning |
| --- | --- | --- |
| `perplexity/model_type` | `causal` | Scorer family: `causal` now, `masked` planned. |
| `perplexity/model` | empty | Rime resource path to the model. Empty disables scoring. |
| `perplexity/device` | `cpu` | `cpu` maps to `gpu_layers=0`; `gpu` maps to `gpu_layers=-1`. |
| `perplexity/gpu_layers` | unset | Advanced override for llama.cpp GPU offload. |
| `perplexity/max_length` | `1024` | Maximum model context length. |
| `perplexity/batch_size` | `32` | Maximum candidate sequences scored in one batch. |
| `perplexity/score_weight` | `1.0` | Multiplier for LM average log probability before adding base score. |
| `perplexity/unknown_token_penalty` | `0.0` | Penalty for unknown or byte-fallback tokens. |
| `perplexity/candidate_types` | `[sentence]` | Candidate types this filter reranks. |
| `perplexity/size` | `2` | Number of reranked candidates promoted to the front. |

## Planned Scorer Families

- `causal`: left-to-right LM perplexity.
- `masked`: masked LM pseudo-log-likelihood.  The interface is already shaped
  for this, but the C++ backend is not implemented yet.
- Encoder-decoder LM is out of scope for now.
