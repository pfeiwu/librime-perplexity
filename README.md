# librime-perplexity

Experimental Rime filter for reranking sentence candidates with local language
model perplexity.

WIP, mostly vibe-coded. Use at your own risk.

This plugin needs multiple sentence candidates from librime. Current stable
librime only outputs one sentence candidate, making this plugin mostly
meaningless; the needed support is in librime commit `9422ca7`
([PR #1164](https://github.com/rime/librime/pull/1164)).

## Requirements

- librime after commit `9422ca7`, built with external plugin support.
- At least one scorer backend:
  - causal LM: llama.cpp headers/shared libraries + a GGUF model.
  - masked LM: ONNX Runtime headers/shared libraries + a BERT-style ONNX
    model and `vocab.txt`.

## Build

Build llama.cpp first if you do not already have it:

```bash
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build -DBUILD_SHARED_LIBS=ON
cmake --build /path/to/llama.cpp/build
```

For GPU acceleration, add the llama.cpp backend option you need, such as
`-DGGML_CUDA=ON`, `-DGGML_VULKAN=ON`, or `-DGGML_METAL=ON`.

Build this plugin in-tree with librime:

```text
librime/plugins/perplexity
```

```bash
cmake -S . -B build \
  -DENABLE_EXTERNAL_PLUGINS=ON \
  -DBUILD_MERGED_PLUGINS=OFF \
  -DPERPLEXITY_LLAMA_CPP_DIR=/path/to/llama.cpp \
  -DPERPLEXITY_LLAMA_CPP_BUILD_DIR=/path/to/llama.cpp/build \
  -DPERPLEXITY_ONNXRUNTIME_DIR=/path/to/onnxruntime

cmake --build build --target rime-perplexity
```

Install `librime-perplexity.{so,dylib,dll}` to your librime plugin directory.
The plugin is dynamically linked to its selected backend libraries, so
llama.cpp/ggml or ONNX Runtime libraries must also be discoverable by the
runtime loader.

`PERPLEXITY_LLAMA_CPP_*` is only needed for `model_type: causal`.
`PERPLEXITY_ONNXRUNTIME_DIR` is only needed for `model_type: masked`.

## Models

### Causal LM

Use a GGUF model:

```yaml
perplexity:
  model_type: causal
  model: models/your-model.gguf
```

### Masked LM

Export a BERT-style masked LM to ONNX, and place `vocab.txt` next to
`model.onnx`:

```text
models/bert-base-chinese/
  model.onnx
  vocab.txt
```

The ONNX graph should expose standard BERT inputs:

```text
input_ids, attention_mask[, token_type_ids] -> logits
```

Schema:

```yaml
perplexity:
  model_type: masked
  model: models/bert-base-chinese/model.onnx
  device: cpu
  batch_size: 32
  score_weight: 60.0
  top_k: 0
```

## Schema

```yaml
engine:
  translators:
    - script_translator
  filters:
    - uniquifier
    - perplexity_ranker

translator:
  max_sentences: 10
  sentence_cutoff_threshold: 1.0

perplexity:
  model_type: causal
  model: models/your-model.gguf
  device: cpu
  batch_size: 32
  cache_size: 0
  score_weight: 60.0
  scan_size: 50
  rank_size: 20
  min_input_size: 0
  top_k: 0
  rank_types:
    - sentence
  history_context_commits: 0
```

`perplexity/model` can be an absolute path or a Rime resource path such as
`models/your-model.gguf`.

## Parameters

| Key | Default | Description |
| --- | --- | --- |
| `perplexity/model_type` | `causal` | `causal` or `masked`. |
| `perplexity/model` | empty | Model path. |
| `perplexity/device` | `cpu` | `cpu` or `gpu`; masked uses ONNX Runtime EP when available. |
| `perplexity/gpu_layers` | unset | llama.cpp GPU layer override; for masked, nonzero means try GPU EP. |
| `perplexity/max_length` | `1024` | Maximum causal context or masked sequence length. |
| `perplexity/batch_size` | `32` | Causal candidate batch size, or masked-token copy batch size. |
| `perplexity/cache_size` | `0` | LRU capacity. Causal: prefix KV cache entries; masked: sentence score cache entries. `0` disables caching. |
| `perplexity/score_weight` | `1.0` | LM score weight. |
| `perplexity/unknown_token_penalty` | `0.0` | Penalty for unknown / byte-fallback tokens. |
| `perplexity/score_prefix` | `，` | Optional text or special token prepended before scoring. |
| `perplexity/score_suffix` | `，` | Optional text or special token appended before scoring. |
| `perplexity/scan_size` | `50` | Number of upstream candidates to scan. |
| `perplexity/rank_size` | `20` | Maximum number of rankable candidates visible to the LM scorer. |
| `perplexity/min_input_size` | `0` | Minimum current input segment length required before scoring; `0` disables this gate. |
| `perplexity/rank_types` | `[sentence]` | Candidate types to rerank. |
| `perplexity/top_k` | `0` | Output cap for reranked candidates kept in rankable slots; `0` keeps all rankable candidates collected by `rank_size`. |
| `perplexity/history_context_commits` | `0` | Recent commit-history records used as scoring context. |

Check Rime logs for `perplexity: loaded causal LM`.

## Notes

- Check the generated schema after deploy; `*.custom.yaml` alone is not proof
  that Rime loaded the intended config.
- For masked GPU, check the log for `ep=CUDA` / `ep=CoreML` / `ep=DirectML`.
  ONNX Runtime GPU builds are tied to specific CUDA versions.
- For causal cache, keep `batch_size + cache_size` within the model's
  llama.cpp sequence limit.
