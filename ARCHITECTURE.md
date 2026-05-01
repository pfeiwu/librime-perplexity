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
  rank_types: [sentence]
  model_type: causal
  model: models/example.gguf
  device: cpu
  max_length: 1024
  batch_size: 32
  cache_size: 0
  score_weight: 60.0
  scan_size: 50
  rank_size: 20
  top_k: 2
  history_context_commits: 0
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
  -> candidates
  -> PerplexityRanker
       scan a bounded upstream candidate window
       collect rankable candidates by Candidate::type()
       build PerplexityInput{context, text, units}
       call PerplexityScorer::Score(batch)
       compute lm_score * score_weight + Phrase::weight()
       fill original rankable slots with up to top_k reranked candidates
       keep scanned non-rankable candidates in place
       append the upstream tail
  -> menu
```

## Internal Boundaries

- `PerplexityRanker`
  - Rime `Filter` implementation.
  - Scans a bounded upstream candidate window and scores candidates whose type
    is rankable.
  - Can use recent commit-history records as scoring context without scoring
    the history text itself.
  - Computes base grammar score from `Phrase::weight()`.
  - Calls a scorer once with a batch of candidate texts.
  - Sorts by `lm_average_logprob * score_weight + base_score`.
  - Fills the original rankable slots with up to `top_k` reranked candidates
    and drops the remaining scored rankable candidates.

- `PerplexityScorer`
  - Model-family abstraction.
  - Input contains both surface text and phrase units.  The current causal and
    masked scorers use the surface text; phrase units are kept for future
    word-aware PLL variants.

- `LlamaCausalScorer`
  - Causal LM backend.
  - Scores causal LM average log probability with llama.cpp.
  - Uses direct per-call batching plus a private cross-call prefix KV cache.
  - Does not use beam reuse, grammar prefilter, or grouped multi-prefix
    batching.

- `OnnxMaskedScorer`
  - Optional backend, enabled by `PERPLEXITY_ONNXRUNTIME_DIR`.
  - Scores masked LM pseudo-log-likelihood (PLL) with ONNX Runtime.
  - Loads standard BERT ONNX graphs with `input_ids`, `attention_mask`, optional
    `token_type_ids`, and `logits` output.
  - Loads `vocab.txt` next to the ONNX model and uses a small WordPiece
    tokenizer path intended for BERT-style Chinese models.
  - For each candidate of length n, creates n masked copies and batches those
    copies through ONNX Runtime.
  - Does not use prefix KV cache; bidirectional attention makes prefix reuse
    invalid.
  - `device: gpu` tries the platform EP when the ONNX Runtime headers expose
    it: Linux CUDA, macOS CoreML, Windows DirectML. Otherwise it falls back to
    CPU and logs a warning.

## Filter Ordering Assumption

`PerplexityRanker` scans a bounded upstream window. Non-rankable candidates in
that window keep their original positions; only rankable slots are filled with
reranked candidates.

Place this filter after candidate deduplication such as `uniquifier`. If a
later filter reorders candidates, it can undo the LM ranking.

Recommended position:

```yaml
engine:
  filters:
    - uniquifier
    - perplexity_ranker
    # ... other filters
```

Filters placed downstream of `perplexity_ranker` operate on the already
reranked output and can apply their own logic.

## Caching

`PerplexityRanker` itself is stateless. The underlying scorer may keep an
optional cache controlled by `perplexity/cache_size`.

### Default: disabled (`cache_size: 0`)

All scoring is fresh. Scores are deterministic across runs and reproducible
regardless of session history. Recommended for evaluation and as a sane
default.

### Causal-only: prefix KV cache (`cache_size: N`)

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

### Masked: sentence-level cache (`cache_size: N`)

Masked LM uses bidirectional attention, so prefix KV reuse is not valid. When
`cache_size > 0`, `OnnxMaskedScorer` instead keeps an LRU cache of complete
sentence scores keyed by candidate text.

This cache has no numerical drift: a cache hit returns the exact normalized
`PerplexityScore` previously computed for the same text. It only avoids
re-tokenizing, rebuilding mask copies, and re-running ONNX inference for
duplicate sentences.

## Parameters

All parameters apply to both `causal` and `masked` model types unless noted.

| Key | Default | Meaning |
| --- | --- | --- |
| `perplexity/model_type` | `causal` | Scorer family: `causal` or `masked`. |
| `perplexity/model` | empty | Rime resource path to the model. Empty disables scoring. |
| `perplexity/device` | `cpu` | `cpu` maps to `gpu_layers=0`; `gpu` maps to `gpu_layers=-1`. |
| `perplexity/gpu_layers` | unset | Advanced llama.cpp GPU offload override; for masked, nonzero means try GPU EP. |
| `perplexity/max_length` | `1024` | Maximum causal context or masked sequence length. |
| `perplexity/batch_size` | `32` | Maximum causal candidate batch size or masked-token copy batch size. |
| `perplexity/cache_size` | `0` | LRU capacity. Causal: prefix KV cache entries; masked: complete sentence score entries. `0` disables caching. |
| `perplexity/score_weight` | `1.0` | Multiplier for LM average log probability before adding base score. |
| `perplexity/unknown_token_penalty` | `0.0` | Penalty for unknown or byte-fallback tokens. |
| `perplexity/score_prefix` | `，` | Optional text or special token prepended before scoring. |
| `perplexity/score_suffix` | `，` | Optional text or special token appended before scoring. |
| `perplexity/scan_size` | `50` | Number of upstream candidates to scan. |
| `perplexity/rank_size` | `20` | Maximum number of rankable candidates visible to the LM scorer. |
| `perplexity/rank_types` | `[sentence]` | Candidate types this filter reranks. |
| `perplexity/top_k` | `2` | Output cap for reranked candidates kept in rankable slots. |
| `perplexity/history_context_commits` | `0` | Recent commit-history records used as scoring context. |

## Scorer Families

- `causal`: left-to-right LM perplexity.
- `masked`: masked LM pseudo-log-likelihood via ONNX Runtime.
- Encoder-decoder LM is out of scope for now.
