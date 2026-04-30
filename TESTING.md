# Testing notes

Use this checklist before trusting benchmark numbers.

1. Build the exact librime tree and plugin backend you want to test.

2. Verify the effective schema, not only `*.custom.yaml`:

   ```bash
   rg -n "perplexity|perplexity_ranker|schema_id" build/rime_ice.schema.yaml
   ```

   The generated schema must show the intended `model_type`, `model`,
   `device`, `score_weight`, and `top_k`.

3. Verify deployment succeeded:

   ```bash
   printf "select schema rime_ice\nexit\n" | ./rime_api_console 2>&1 \
     | rg -i "deploy|failure|schema|error|warn"
   ```

   Do not benchmark after `deploy failure`.

4. Verify the schema can produce candidates:

   ```bash
   printf "select schema rime_ice\nset input <known-valid-code>\nprint candidate list\nexit\n" \
     | ./rime_api_console
   ```

5. Verify the backend actually loaded.

   For causal LM, logs should mention llama.cpp/model loading. For masked LM,
   logs should mention the ONNX model and execution provider. If logs are not
   visible, verify through a scorer probe before running the full benchmark.

6. Only then run the corpus benchmark. Treat `0/N`, empty candidates, or
   implausibly fast masked-LM runs as invalid until the above checks pass.

## Known traps

- Rime rebuilds `build/*.schema.yaml` on deploy. Check the generated schema
  after each config change; do not trust an old generated file.

- Run `rime_api_console` from the intended user data directory. In local tests
  this is usually `librime/build/bin`; launching from a different cwd can load
  a different schema/config set.

- Match the benchmark binary and plugin build. If the console or benchmark uses
  a librime build that was not compiled with `PERPLEXITY_ONNXRUNTIME_DIR`, a
  masked model will log:

  ```text
  perplexity: masked LM scorer requires ONNX Runtime
  ```

- `device: gpu` for masked LM only works if the linked ONNX Runtime package has
  a compatible GPU execution provider. ONNX Runtime GPU 1.25.1 expects CUDA 12
  libraries (`libcudart.so.12`, `libcublas.so.12`, `libcufft.so.11`). On a CUDA
  13 system it will not link unless you use a CUDA-13-compatible ORT build.

- Do not directly link optional ONNX provider libraries such as
  `libonnxruntime_providers_cuda.so` into test binaries. Link `libonnxruntime`
  and `libonnxruntime_providers_shared`; let ONNX Runtime load the CUDA provider
  at runtime.

- Confirm GPU execution from the scorer log, not from the config alone:

  ```text
  perplexity: loaded masked LM: ..., ep=CUDA, ...
  ```

- Causal `cache_size` consumes llama.cpp sequence slots. Effective
  `n_seq_max = batch_size + cache_size`, and some GPT-2 GGUF models reject
  values above 256. With `batch_size: 32`, `cache_size: 256` can fail with:

  ```text
  llama_init_from_model: failed to initialize the context: n_seq_max must be <= 256
  ```

  Use a smaller cache, for example `cache_size: 128`.

- Some HF models combine a GPT-style causal architecture with a T5 /
  SentencePiece tokenizer. Current llama.cpp conversion/tokenization can be
  fragile for that mix. Verified examples:

  ```text
  rinna/japanese-gpt2-small
  rinna/japanese-gpt-neox-small
  ```

  `llama-tokenize` may throw `std::out_of_range: unordered_map::at` for
  ordinary-looking punctuation such as `，`, and current
  `convert_hf_to_gguf.py` can fail with:

  ```text
  BPE pre-tokenizer was not recognized
  ```

  Before using such a model in Rime, test the GGUF outside the IME:

  ```bash
  llama-tokenize -m model.gguf -p "こんにちは"
  llama-tokenize -m model.gguf -p "今日は良い天気です，"
  llama-tokenize -m model.gguf -p "枯れたはずの枝に積もった雪，"
  ```
