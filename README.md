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
- llama.cpp headers and shared libraries.
- A GGUF causal LM model.

## Build

Build llama.cpp first if you do not already have it:

```bash
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build -DBUILD_SHARED_LIBS=ON
cmake --build /path/to/llama.cpp/build
```

For GPU acceleration, add the llama.cpp backend option you need, such as CUDA,
Vulkan, or Metal.

Build this plugin in-tree with librime:

```text
librime/plugins/perplexity
```

```bash
cmake -S . -B build \
  -DENABLE_EXTERNAL_PLUGINS=ON \
  -DPERPLEXITY_LLAMA_CPP_DIR=/path/to/llama.cpp \
  -DPERPLEXITY_LLAMA_CPP_BUILD_DIR=/path/to/llama.cpp/build

cmake --build build --target rime-perplexity
```

Install `librime-perplexity.{so,dylib,dll}` to your librime plugin directory.

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
  size: 2
  candidate_types:
    - sentence
```

`perplexity/model` can be an absolute path or a Rime resource path such as
`models/your-model.gguf`.

## Parameters

| Key | Default | Description |
| --- | --- | --- |
| `perplexity/model_type` | `causal` | `causal` now; `masked` is reserved for MLM pseudo-perplexity. |
| `perplexity/model` | empty | Model path. |
| `perplexity/device` | `cpu` | `cpu` or `gpu`. |
| `perplexity/gpu_layers` | unset | llama.cpp GPU layer override. |
| `perplexity/max_length` | `1024` | Context length. |
| `perplexity/batch_size` | `32` | Scoring batch size. |
| `perplexity/cache_size` | `0` | Prefix KV cache size; `0` disables it. |
| `perplexity/score_weight` | `1.0` | LM score weight. |
| `perplexity/unknown_token_penalty` | `0.0` | Penalty for unknown / byte-fallback tokens. |
| `perplexity/candidate_types` | `[sentence]` | Candidate types to rerank. |
| `perplexity/size` | `2` | Number of reranked candidates to promote. |

Check Rime logs for `perplexity: loaded causal LM`.
