// SPDX-License-Identifier: BSD-3-Clause

#include "scorer.h"

#include "bert_wordpiece_tokenizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <fstream>
#include <limits>
#include <list>
#include <numeric>
#include <unordered_map>

#include <rime/common.h>

#ifdef RIME_PERPLEXITY_ENABLE_ONNX
#include <onnxruntime_cxx_api.h>

#if defined(__has_include)
#if __has_include(<onnxruntime/core/providers/cuda/cuda_provider_factory.h>)
#include <onnxruntime/core/providers/cuda/cuda_provider_factory.h>
#define RIME_PERPLEXITY_HAVE_ORT_CUDA 1
#endif
#if __has_include(<onnxruntime/core/providers/coreml/coreml_provider_factory.h>)
#include <onnxruntime/core/providers/coreml/coreml_provider_factory.h>
#define RIME_PERPLEXITY_HAVE_ORT_COREML 1
#endif
#if __has_include(<onnxruntime/core/providers/dml/dml_provider_factory.h>)
#include <onnxruntime/core/providers/dml/dml_provider_factory.h>
#define RIME_PERPLEXITY_HAVE_ORT_DML 1
#endif
#endif
#endif

namespace rime {

namespace {

#ifdef RIME_PERPLEXITY_ENABLE_ONNX

using int64 = int64_t;

struct MaskTask {
  size_t input_index = 0;
  int mask_pos = 0;
  int64 target_id = 0;
  bool is_unk = false;
  vector<int64> ids;
};

static bool FileExists(const string& path) {
  std::ifstream stream(path);
  return stream.good();
}

static string DirName(const string& path) {
  const size_t pos = path.find_last_of("/\\");
  return pos == string::npos ? string() : path.substr(0, pos);
}

static string JoinPath(const string& dir, const string& file) {
  if (dir.empty())
    return file;
  const char last = dir[dir.size() - 1];
  if (last == '/' || last == '\\')
    return dir + file;
  return dir + "/" + file;
}

static string ResolveModelPath(const string& path) {
  if (FileExists(path))
    return path;
  const string model = JoinPath(path, "model.onnx");
  if (FileExists(model))
    return model;
  return path;
}

static string ResolveVocabPath(const string& model_path) {
  const string dir = DirName(model_path);
  const string vocab = JoinPath(dir, "vocab.txt");
  if (FileExists(vocab))
    return vocab;
  const string parent = DirName(dir);
  const string parent_vocab = JoinPath(parent, "vocab.txt");
  if (FileExists(parent_vocab))
    return parent_vocab;
  return vocab;
}

class OnnxMaskedScorer : public PerplexityScorer {
 public:
  explicit OnnxMaskedScorer(const PerplexityScorerOptions& options)
      : env_(ORT_LOGGING_LEVEL_WARNING, "rime-perplexity"),
        batch_size_(std::max(1, options.batch_size)),
        max_length_(std::max(3, options.max_length)),
        unknown_token_penalty_(options.unknown_token_penalty),
        cache_capacity_(static_cast<size_t>(std::max(0, options.cache_size))) {
    memory_info_ = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    run_options_ = std::make_unique<Ort::RunOptions>(nullptr);
    const string model_path = ResolveModelPath(options.model_path);
    const string vocab_path = ResolveVocabPath(model_path);
    if (!tokenizer_.Load(vocab_path))
      return;

    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    ep_name_ = ConfigureExecutionProvider(options.gpu_layers);

    try {
      session_ = std::make_unique<Ort::Session>(
          env_, model_path.c_str(), session_options_);
      InitNames();
    } catch (const Ort::Exception& e) {
      LOG(ERROR) << "perplexity: failed to load masked LM: " << model_path
                 << ": " << e.what();
      session_.reset();
      return;
    }

    if (!HasInput("input_ids") || !HasInput("attention_mask")) {
      LOG(ERROR) << "perplexity: masked LM ONNX model must expose input_ids "
                    "and attention_mask inputs";
      session_.reset();
      return;
    }
    if (output_names_.empty()) {
      LOG(ERROR) << "perplexity: masked LM ONNX model has no outputs";
      session_.reset();
      return;
    }
    LOG(INFO) << "perplexity: loaded masked LM: " << model_path
              << ", vocab=" << vocab_path
              << ", ep=" << ep_name_
              << ", batch_size=" << batch_size_
              << ", max_length=" << max_length_
              << ", cache_size=" << cache_capacity_;
  }

  bool Ready() const override { return static_cast<bool>(session_); }

  vector<PerplexityScore> Score(
      const vector<PerplexityInput>& inputs) override {
    vector<PerplexityScore> scores(inputs.size());
    if (!Ready() || inputs.empty())
      return scores;

    vector<size_t> miss_indices;
    miss_indices.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (cache_capacity_ > 0 && CacheGet(CacheKey(inputs[i]), &scores[i]))
        continue;
      miss_indices.push_back(i);
    }

    vector<MaskTask> tasks;
    for (size_t i : miss_indices) {
      AddTasksForInput(inputs[i], i, &tasks);
    }
    for (size_t begin = 0; begin < tasks.size(); begin += batch_size_) {
      const size_t end = std::min(tasks.size(), begin + batch_size_);
      ScoreTaskBatch(tasks, begin, end, &scores);
    }
    for (size_t i : miss_indices) {
      PerplexityScore& score = scores[i];
      if (score.token_count > 0) {
        score.average_logprob /= score.token_count;
      }
      if (cache_capacity_ > 0)
        CachePut(CacheKey(inputs[i]), score);
    }
    return scores;
  }

 private:
  using CacheList = std::list<std::pair<string, PerplexityScore>>;

  static string CacheKey(const PerplexityInput& input) {
    return input.context + "\x1f" + input.text;
  }

  bool CacheGet(const string& key, PerplexityScore* out) {
    auto it = cache_index_.find(key);
    if (it == cache_index_.end())
      return false;
    cache_lru_.splice(cache_lru_.begin(), cache_lru_, it->second);
    *out = cache_lru_.begin()->second;
    return true;
  }

  void CachePut(const string& key, const PerplexityScore& score) {
    if (cache_capacity_ == 0)
      return;
    auto it = cache_index_.find(key);
    if (it != cache_index_.end()) {
      it->second->second = score;
      cache_lru_.splice(cache_lru_.begin(), cache_lru_, it->second);
      return;
    }
    cache_lru_.emplace_front(key, score);
    cache_index_[cache_lru_.front().first] = cache_lru_.begin();
    while (cache_lru_.size() > cache_capacity_) {
      cache_index_.erase(cache_lru_.back().first);
      cache_lru_.pop_back();
    }
  }

  string ConfigureExecutionProvider(int gpu_layers) {
    if (gpu_layers == 0)
      return "CPU";
#if defined(__APPLE__) && defined(RIME_PERPLEXITY_HAVE_ORT_COREML)
    try {
      OrtSessionOptionsAppendExecutionProvider_CoreML(session_options_, 0);
      return "CoreML";
    } catch (...) {
      LOG(WARNING) << "perplexity: failed to enable CoreML EP; using CPU";
    }
#elif defined(_WIN32) && defined(RIME_PERPLEXITY_HAVE_ORT_DML)
    try {
      OrtSessionOptionsAppendExecutionProvider_DML(session_options_, 0);
      return "DirectML";
    } catch (...) {
      LOG(WARNING) << "perplexity: failed to enable DirectML EP; using CPU";
    }
#elif defined(RIME_PERPLEXITY_HAVE_ORT_CUDA)
    if (OrtStatus* status =
            OrtSessionOptionsAppendExecutionProvider_CUDA(session_options_, 0)) {
      LOG(WARNING) << "perplexity: failed to enable CUDA EP: "
                   << Ort::GetApi().GetErrorMessage(status)
                   << "; using CPU";
      Ort::GetApi().ReleaseStatus(status);
    } else {
      return "CUDA";
    }
#else
    LOG(WARNING) << "perplexity: ONNX GPU EP headers are unavailable; using CPU";
#endif
    return "CPU";
  }

  void InitNames() {
    Ort::AllocatorWithDefaultOptions allocator;
    input_names_.clear();
    input_name_storage_.clear();
    for (size_t i = 0; i < session_->GetInputCount(); ++i) {
      auto name = session_->GetInputNameAllocated(i, allocator);
      input_name_storage_.push_back(name.get());
    }
    for (const auto& name : input_name_storage_)
      input_names_.push_back(name.c_str());

    output_names_.clear();
    output_name_storage_.clear();
    for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
      auto name = session_->GetOutputNameAllocated(i, allocator);
      output_name_storage_.push_back(name.get());
    }
    for (const auto& name : output_name_storage_)
      output_names_.push_back(name.c_str());
  }

  bool HasInput(const string& name) const {
    return std::find(input_name_storage_.begin(), input_name_storage_.end(),
                     name) != input_name_storage_.end();
  }

  void AddTasksForInput(const PerplexityInput& input,
                        size_t input_index,
                        vector<MaskTask>* tasks) const {
    const int max_content_tokens = std::max(1, max_length_ - 2);
    BertTokenizedText context_tokens;
    BertTokenizedText target_tokens;
    try {
      context_tokens = tokenizer_.Tokenize(input.context, max_content_tokens);
      target_tokens = tokenizer_.Tokenize(input.text, max_content_tokens);
    } catch (const std::exception& e) {
      LOG(ERROR) << "perplexity: masked tokenize threw on text=\""
                 << input.text.substr(0, 64) << "\": " << e.what();
      return;
    }
    if (target_tokens.tokens.empty())
      return;
    const size_t context_limit =
        static_cast<size_t>(std::max(0, max_content_tokens -
                                            static_cast<int>(
                                                target_tokens.tokens.size())));
    if (context_tokens.tokens.size() > context_limit) {
      const size_t drop = context_tokens.tokens.size() - context_limit;
      context_tokens.tokens.erase(
          context_tokens.tokens.begin(), context_tokens.tokens.begin() + drop);
    }
    vector<int64> base;
    base.reserve(context_tokens.tokens.size() + target_tokens.tokens.size() + 2);
    base.push_back(tokenizer_.cls_id());
    base.insert(base.end(), context_tokens.tokens.begin(),
                context_tokens.tokens.end());
    const size_t target_start = base.size();
    base.insert(base.end(), target_tokens.tokens.begin(),
                target_tokens.tokens.end());
    base.push_back(tokenizer_.sep_id());
    for (size_t pos = target_start; pos + 1 < base.size(); ++pos) {
      MaskTask task;
      task.input_index = input_index;
      task.mask_pos = static_cast<int>(pos);
      task.target_id = base[pos];
      task.is_unk = task.target_id == tokenizer_.unk_id();
      task.ids = base;
      task.ids[pos] = tokenizer_.mask_id();
      tasks->push_back(std::move(task));
    }
  }

  void ScoreTaskBatch(const vector<MaskTask>& tasks,
                      size_t begin,
                      size_t end,
                      vector<PerplexityScore>* scores) {
    const size_t batch = end - begin;
    int64 seq_len = 0;
    for (size_t i = begin; i < end; ++i) {
      seq_len = std::max(seq_len, static_cast<int64>(tasks[i].ids.size()));
    }

    const size_t tensor_size = batch * static_cast<size_t>(seq_len);
    input_ids_.assign(tensor_size, tokenizer_.pad_id());
    attention_mask_.assign(tensor_size, 0);
    token_type_ids_.assign(tensor_size, 0);
    for (size_t row = 0; row < batch; ++row) {
      const auto& ids = tasks[begin + row].ids;
      for (size_t col = 0; col < ids.size(); ++col) {
        const size_t offset = row * seq_len + col;
        input_ids_[offset] = ids[col];
        attention_mask_[offset] = 1;
      }
    }

    std::array<int64, 2> shape = {
        static_cast<int64>(batch),
        seq_len,
    };
    vector<Ort::Value> values;
    vector<const char*> names;
    names.push_back("input_ids");
    values.push_back(Ort::Value::CreateTensor<int64>(
        *memory_info_, input_ids_.data(), input_ids_.size(), shape.data(),
        shape.size()));
    names.push_back("attention_mask");
    values.push_back(Ort::Value::CreateTensor<int64>(
        *memory_info_, attention_mask_.data(), attention_mask_.size(), shape.data(),
        shape.size()));
    if (HasInput("token_type_ids")) {
      names.push_back("token_type_ids");
      values.push_back(Ort::Value::CreateTensor<int64>(
          *memory_info_, token_type_ids_.data(), token_type_ids_.size(), shape.data(),
          shape.size()));
    }

    try {
      auto outputs = session_->Run(*run_options_, names.data(),
                                   values.data(), values.size(),
                                   output_names_.data(), 1);
      if (outputs.empty() || !outputs[0].IsTensor())
        return;
      const float* logits = outputs[0].GetTensorData<float>();
      auto info = outputs[0].GetTensorTypeAndShapeInfo();
      vector<int64> out_shape = info.GetShape();
      if (out_shape.size() != 3)
        return;
      const int64 out_seq_len = out_shape[1];
      const int64 vocab_size = out_shape[2];
      for (size_t row = 0; row < batch; ++row) {
        const MaskTask& task = tasks[begin + row];
        if (task.target_id < 0 || task.target_id >= vocab_size ||
            task.mask_pos >= out_seq_len) {
          continue;
        }
        const float* row_logits =
            logits + (row * out_seq_len + task.mask_pos) * vocab_size;
        const double logprob = LogSoftmaxAt(row_logits, vocab_size,
                                            task.target_id);
        PerplexityScore& score = (*scores)[task.input_index];
        if (score.token_count == 0)
          score.average_logprob = 0.0;
        score.average_logprob +=
            logprob - (task.is_unk ? unknown_token_penalty_ : 0.0);
        ++score.token_count;
      }
    } catch (const Ort::Exception& e) {
      LOG(ERROR) << "perplexity: masked LM inference failed: " << e.what();
    }
  }

  static double LogSoftmaxAt(const float* logits,
                             int64 vocab_size,
                             int64 target_id) {
    double max_logit = -std::numeric_limits<double>::infinity();
    for (int64 i = 0; i < vocab_size; ++i) {
      max_logit = std::max(max_logit, static_cast<double>(logits[i]));
    }
    double sum = 0.0;
    for (int64 i = 0; i < vocab_size; ++i) {
      sum += std::exp(static_cast<double>(logits[i]) - max_logit);
    }
    return static_cast<double>(logits[target_id]) - max_logit -
           std::log(sum);
  }

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::MemoryInfo> memory_info_;
  std::unique_ptr<Ort::RunOptions> run_options_;
  BertWordPieceTokenizer tokenizer_;
  vector<string> input_name_storage_;
  vector<string> output_name_storage_;
  vector<const char*> input_names_;
  vector<const char*> output_names_;
  int batch_size_ = 32;
  int max_length_ = 512;
  double unknown_token_penalty_ = 0.0;
  size_t cache_capacity_ = 0;
  vector<int64> input_ids_;
  vector<int64> attention_mask_;
  vector<int64> token_type_ids_;
  CacheList cache_lru_;
  std::unordered_map<string, CacheList::iterator> cache_index_;
  string ep_name_ = "CPU";
};

#endif  // RIME_PERPLEXITY_ENABLE_ONNX

}  // namespace

std::unique_ptr<PerplexityScorer> CreateOnnxMaskedScorer(
    const PerplexityScorerOptions& options) {
#ifdef RIME_PERPLEXITY_ENABLE_ONNX
  return std::make_unique<OnnxMaskedScorer>(options);
#else
  LOG(ERROR) << "perplexity: masked LM scorer requires ONNX Runtime; rebuild "
                "with PERPLEXITY_ONNXRUNTIME_DIR";
  return nullptr;
#endif
}

}  // namespace rime
