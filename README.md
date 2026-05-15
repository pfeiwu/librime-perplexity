# librime-perplexity

Experimental Rime plugin providing a perplexity-based filter for reranking
candidates with a local language model.

WIP, mostly vibe-coded. Use at your own risk.

## Requirements

- librime built with external plugin support.
- At least one scorer backend:
  - causal LM: llama.cpp headers/shared libraries + a GGUF model.
  - masked LM: ONNX Runtime headers/shared libraries + a BERT-style ONNX
    model and `vocab.txt`.

## Build

Install the plugin from the librime root with `install-plugins.sh`. The action
script prepares llama.cpp and ONNX Runtime under `plugins/perplexity/thirdparty`,
where the plugin's CMake auto-discovers them.

```bash
cd /path/to/librime
bash install-plugins.sh run=plugins/perplexity/action-install.sh pfeiwu/librime-perplexity
```

Defaults: macOS gets Metal automatically; Linux / Windows build CPU
llama.cpp; ONNX Runtime is the prebuilt CPU tarball on all platforms.

To pick a different backend, set environment variables before running
`install-plugins.sh`:

| Variable                  | Values                                      | Default                           |
| ------------------------- | ------------------------------------------- | --------------------------------- |
| `PERPLEXITY_LLAMA_BACKEND` | `cpu`, `metal`, `cuda`, `vulkan`, `hip`     | `metal` on macOS, `cpu` elsewhere |
| `PERPLEXITY_ORT_BACKEND`   | `cpu`, `cuda`                               | `cpu`                             |
| `PERPLEXITY_LLAMA_REPO`    | git URL                                     | upstream `ggml-org/llama.cpp`     |
| `PERPLEXITY_LLAMA_REF`     | branch or tag                               | `master`                          |
| `PERPLEXITY_ORT_VERSION`   | release version                             | `1.20.1`                          |

GPU backends require their toolchain to be installed (CUDA Toolkit, Vulkan
SDK, ROCm) — this is on you, the action script just passes the right
`-DGGML_*=ON` flag to llama.cpp.

## Install

### Linux

Install into the system librime plugin directory:

```bash
cd /path/to/librime
make install
sudo install -m 755 dist/lib/rime-plugins/librime-perplexity.so \
  /usr/lib/rime-plugins/librime-perplexity.so
sudo mkdir -p /usr/lib/rime-plugins/perplexity
sudo rsync -a dist/lib/rime-plugins/perplexity/ \
  /usr/lib/rime-plugins/perplexity/
```

### macOS

Install into Squirrel's bundled librime plugin directory:

```bash
cd /path/to/librime
make install
sudo install -m 755 dist/lib/rime-plugins/librime-perplexity.dylib \
  "/Library/Input Methods/Squirrel.app/Contents/Frameworks/rime-plugins/librime-perplexity.dylib"
sudo mkdir -p \
  "/Library/Input Methods/Squirrel.app/Contents/Frameworks/rime-plugins/perplexity"
sudo rsync -a dist/lib/rime-plugins/perplexity/ \
  "/Library/Input Methods/Squirrel.app/Contents/Frameworks/rime-plugins/perplexity/"
```

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
  score_weight: 0.5
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
  score_weight: 0.5
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
| `perplexity/score_weight` | `1.0` | Blend ratio in `[0.0, 1.0]`: `0.0` keeps base ordering, `1.0` uses LM ordering among candidates with the same input span. Longer input spans still rank first. |
| `perplexity/unknown_token_penalty` | `0.0` | Penalty for unknown / byte-fallback tokens. |
| `perplexity/score_prefix` | `，` | Optional text or special token prepended before scoring. |
| `perplexity/score_suffix` | `，` | Optional text or special token appended before scoring. |
| `perplexity/scan_size` | `50` | Number of upstream candidates to scan. |
| `perplexity/rank_size` | `20` | Maximum number of rankable candidates visible to the LM scorer. |
| `perplexity/min_input_size` | `0` | Minimum current input segment length required before scoring; `0` disables this gate. |
| `perplexity/rank_types` | `[sentence]` | Candidate types to rerank. |
| `perplexity/top_k` | `0` | Output cap for reranked candidates kept in rankable slots; `0` keeps all rankable candidates collected by `rank_size`. |
| `perplexity/history_context_commits` | `0` | Recent commit-history records used as scoring context. |

Check Rime logs for `perplexity: loaded new cached causal LM` and
`perplexity: reusing cached causal LM`.

## Notes

- Reranking multiple sentence candidates requires librime commit `9422ca7` or
  newer ([PR #1164](https://github.com/rime/librime/pull/1164)).
- For masked GPU, check the log for `ep=CUDA` / `ep=CoreML` / `ep=DirectML`.
  ONNX Runtime GPU builds are tied to specific CUDA versions.
- For causal cache, keep `batch_size + cache_size` within the model's
  llama.cpp sequence limit.
- Multiple causal ranker filter instances with the same model path and
  `gpu_layers` share one loaded llama model in the process. Each instance
  still owns its own mutable llama context and prefix KV cache.
