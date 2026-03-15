/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
// Implementation of helper utilities for creating and configuring LLM runners

#include <executorch/extension/llm/runner/image_prefiller.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/multimodal_decoder_runner.h>
#include <executorch/extension/llm/runner/multimodal_prefiller.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/io_manager/runner_managed_cache_io_manager.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/llm/runner/text_prefiller.h>
#include <executorch/extension/llm/runner/text_token_generator.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <pytorch/tokenizers/sentencepiece.h>
#include <pytorch/tokenizers/tekken.h>
#include <pytorch/tokenizers/tiktoken.h>

namespace executorch::extension::llm {

using ::executorch::extension::Module;
using ::executorch::runtime::Error;

std::unique_ptr<tokenizers::Tokenizer> load_tokenizer(
    const std::string& tokenizer_path,
    std::unique_ptr<std::vector<std::string>> special_tokens,
    std::optional<std::string> pattern,
    size_t bos_token_index,
    size_t eos_token_index) {
  runtime::runtime_init();
  auto tekken_tokenizer = std::make_unique<tokenizers::Tekken>();
  // Prevent the case where tekken tokenizer accidentally successfully loads a
  // HuggingFace tokenizer, which is also .json.
  static constexpr std::string_view tekken_name = "tekken.json";
  if (tokenizer_path.size() >= tekken_name.size() &&
      tokenizer_path.rfind(tekken_name) ==
          tokenizer_path.size() - tekken_name.size()) {
    if (tekken_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
      ET_LOG(Info, "Loaded tekken tokenizer");
      return tekken_tokenizer;
    }
  }
  auto json_tokenizer = std::make_unique<tokenizers::HFTokenizer>();
  if (json_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded json tokenizer");
    return json_tokenizer;
  }
  std::unique_ptr<::tokenizers::Tiktoken> tiktoken_tokenizer;
  if (special_tokens != nullptr && !pattern.has_value()) {
    tiktoken_tokenizer = std::make_unique<::tokenizers::Tiktoken>(
        std::move(special_tokens), bos_token_index, eos_token_index);
  } else if (special_tokens != nullptr && pattern.has_value()) {
    tiktoken_tokenizer = std::make_unique<::tokenizers::Tiktoken>(
        pattern.value(),
        std::move(special_tokens),
        bos_token_index,
        eos_token_index);
  } else {
    tiktoken_tokenizer = std::make_unique<::tokenizers::Tiktoken>();
  }
  if (tiktoken_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded TikToken tokenizer");
    return tiktoken_tokenizer;
  }

  auto sp_tokenizer = std::make_unique<::tokenizers::SPTokenizer>();
  if (sp_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded Sentencepiece tokenizer");
    return sp_tokenizer;
  }

  auto bpe_tokenizer = std::make_unique<::tokenizers::Llama2cTokenizer>();
  if (bpe_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded BPE tokenizer");
    return bpe_tokenizer;
  }

  return nullptr;
}

::executorch::runtime::Result<std::unordered_map<std::string, int64_t>>
get_llm_metadata(tokenizers::Tokenizer* tokenizer, Module* module) {
  // Initialize metadata with default values
  std::unordered_map<std::string, int64_t> metadata({
      {llm::kEnableDynamicShape, false},
      {llm::kMaxSeqLen, 128},
      {llm::kMaxContextLen, 128},
      {llm::kUseKVCache, true},
      {llm::kUseSDPAWithKVCache, false},
      {llm::kRunnerManagedCache, false},
  });

  // Read metadata from the model
  auto method_names_result = module->method_names();
  if (method_names_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed reading method names");
    return ::executorch::runtime::Error::InvalidArgument;
  }
  const auto& method_names = method_names_result.get();

  // Error out if the max seq len metadata method is not present, since
  // it is hard to figure out from just the .pte itself.
  if (!method_names.count(llm::kMaxSeqLen)) {
    ET_LOG(
        Error,
        "Required metadata method %s not found in model",
        llm::kMaxSeqLen);
    return ::executorch::runtime::Error::InvalidArgument;
  }

  for (auto& pair : metadata) {
    const auto& method_name = pair.first;
    auto& value = pair.second;

    if (method_names.count(method_name)) {
      auto get_result = module->get(method_name);
      value = get_result.get().toScalar().to<decltype(metadata)::mapped_type>();
    } else {
      ET_LOG(
          Info,
          "Method %s not found, using the default value %" PRId64,
          method_name.c_str(),
          value);
    }
    ET_LOG(Info, "Metadata: %s = %" PRId64, method_name.c_str(), value);
  }

  // If kMaxContextLen method not found but kMaxSeqLen is
  // available, set kMaxContextLen to the value of kMaxSeqLen.
  if (!method_names.count(llm::kMaxContextLen) &&
      method_names.count(llm::kMaxSeqLen)) {
    metadata[llm::kMaxContextLen] = metadata[llm::kMaxSeqLen];
    ET_LOG(
        Info,
        "Setting kMaxContextLen to kMaxSeqLen value: %" PRId64,
        metadata[llm::kMaxContextLen]);
  }

  // Set tokenizer-related metadata
  metadata[llm::kBosId] = tokenizer->bos_tok();
  metadata[llm::kVocabSize] = tokenizer->vocab_size();
  return metadata;
}

std::unordered_set<uint64_t> get_eos_ids(
    tokenizers::Tokenizer* tokenizer,
    Module* module) {
  std::unordered_set<uint64_t> eos_ids = {tokenizer->eos_tok()};
  // Get EOS IDs if available
  auto method_names_result = module->method_names();
  if (method_names_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed reading method names");
    return eos_ids;
  }
  const auto& method_names = method_names_result.get();

  if (method_names.count(llm::kEosIds)) {
    eos_ids.clear();
    auto execute_result = module->execute(llm::kEosIds);
    if (execute_result.error() != Error::Ok) {
      ET_LOG(Error, "Failed to execute %s", llm::kEosIds);
      return eos_ids;
    }
    for (const auto& eos_id : execute_result.get()) {
      auto value = eos_id.toScalar().to<int64_t>();
      eos_ids.emplace(value);
      ET_LOG(Info, "eos_id = %" PRId64, value);
    }
  }
  return eos_ids;
}

std::unique_ptr<TextLLMRunner> create_text_llm_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::optional<const std::string> data_path,
    float temperature,
    const std::string& method_name,
    Module::LoadMode load_mode,
    const std::string& prefill_method_name) {
  if (data_path.has_value()) {
    std::vector<std::string> data_files;
    data_files.push_back(data_path.value());
    return create_text_llm_runner(
        model_path,
        std::move(tokenizer),
        std::move(data_files),
        temperature,
        nullptr,
        method_name,
        load_mode,
        prefill_method_name);
  }
  return create_text_llm_runner(
      model_path,
      std::move(tokenizer),
      std::vector<std::string>(),
      temperature,
      nullptr,
      method_name,
      load_mode,
      prefill_method_name);
}

std::unique_ptr<TextLLMRunner> create_text_llm_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::vector<std::string> data_files,
    float temperature,
    std::unique_ptr<::executorch::runtime::EventTracer> event_tracer,
    const std::string& method_name,
    Module::LoadMode load_mode,
    const std::string& prefill_method_name) {
  // Sanity check tokenizer
  if (!tokenizer || !tokenizer->is_loaded()) {
    ET_LOG(Error, "Tokenizer is null or not loaded");
    return nullptr;
  }

  // Create the Module
  std::unique_ptr<Module> module;
  if (data_files.size() > 0) {
    module = std::make_unique<Module>(
        model_path, data_files, load_mode, std::move(event_tracer));
  } else {
    module = std::make_unique<Module>(
        model_path, load_mode, std::move(event_tracer));
  }

  // Get metadata from Module
  ET_LOG(Info, "Reading metadata from model");
  auto metadata_result = llm::get_llm_metadata(tokenizer.get(), module.get());
  if (metadata_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to get metadata from model");
    return nullptr;
  }
  auto metadata = metadata_result.get();

  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      llm::get_eos_ids(tokenizer.get(), module.get()));

  // Create IOManager
  std::unique_ptr<IOManager> io_manager;
  if (metadata.at(kRunnerManagedCache)) {
    // Runner-managed cache: the model outputs K/V projections, the runner
    // writes them into the cache. No shared mutable buffers needed.
    auto n_layers = metadata.at(kMaxContextLen);  // placeholder; see below
    // We need n_layers, n_kv_heads, head_dim from the decode method's input
    // metadata. The k_cache input (index 2) has shape
    // (n_layers, 1, n_kv_heads, max_seq_len, head_dim).
    auto decode_method_name = method_name;
    auto dm = module->method_meta(decode_method_name);
    if (dm.ok() && dm->num_inputs() >= 4) {
      auto k_meta = dm->input_tensor_meta(2);
      if (k_meta.ok()) {
        auto sizes = k_meta->sizes();
        // sizes: [n_layers, batch, n_kv_heads, max_seq_len, head_dim]
        n_layers = sizes[0];
        auto n_kv_heads = sizes[2];
        auto max_seq = sizes[3];
        auto head_dim = sizes[4];
        auto dtype = k_meta->scalar_type();
        io_manager = std::make_unique<RunnerManagedCacheIOManager>(
            *module, n_layers, n_kv_heads, max_seq, head_dim, dtype);
      }
    }
    if (!io_manager) {
      ET_LOG(Error,
             "runner_managed_cache set but could not read k_cache metadata "
             "from decode method. Falling back to default IOManager.");
      io_manager = std::make_unique<IOManager>(*module);
    }
  } else {
    io_manager = std::make_unique<IOManager>(*module);
  }

  // When using separate prefill/decode methods with model-managed cache,
  // pre-load both with shared planned memory so they share the same KV cache
  // buffers. Runner-managed cache doesn't need shared buffers.
  std::unique_ptr<TextLLMRunner::SharedMethodMemory> shared_method_memory;
  if (!prefill_method_name.empty() && !metadata.at(kRunnerManagedCache)) {
    ET_LOG(
        Info,
        "Pre-loading methods with shared memory: %s, %s",
        prefill_method_name.c_str(),
        method_name.c_str());

    auto prefill_meta = module->method_meta(prefill_method_name);
    auto decode_meta = module->method_meta(method_name);
    if (!prefill_meta.ok() || !decode_meta.ok()) {
      ET_LOG(Error, "Failed to get method metadata for shared memory setup");
      return nullptr;
    }

    size_t prefill_bufs = prefill_meta->num_memory_planned_buffers();
    size_t decode_bufs = decode_meta->num_memory_planned_buffers();
    size_t num_buffers = std::max(prefill_bufs, decode_bufs);

    shared_method_memory =
        std::make_unique<TextLLMRunner::SharedMethodMemory>();
    shared_method_memory->buffers.reserve(num_buffers);
    shared_method_memory->spans.reserve(num_buffers);

    for (size_t i = 0; i < num_buffers; ++i) {
      int64_t pf_size =
          (i < prefill_bufs)
          ? prefill_meta->memory_planned_buffer_size(i).get()
          : 0;
      int64_t dc_size =
          (i < decode_bufs)
          ? decode_meta->memory_planned_buffer_size(i).get()
          : 0;
      size_t max_size = static_cast<size_t>(std::max(pf_size, dc_size));
      shared_method_memory->buffers.emplace_back(max_size);
      shared_method_memory->spans.emplace_back(
          shared_method_memory->buffers.back().data(), max_size);
    }

    shared_method_memory->allocator =
        std::make_unique<::executorch::runtime::HierarchicalAllocator>(
            ::executorch::runtime::Span<::executorch::runtime::Span<uint8_t>>(
                shared_method_memory->spans.data(),
                shared_method_memory->spans.size()));

    auto err = module->load_method(
        prefill_method_name, shared_method_memory->allocator.get());
    if (err != Error::Ok) {
      ET_LOG(Error, "Failed to load prefill method with shared memory");
      return nullptr;
    }
    err = module->load_method(
        method_name, shared_method_memory->allocator.get());
    if (err != Error::Ok) {
      ET_LOG(Error, "Failed to load decode method with shared memory");
      return nullptr;
    }
  }

  // Create text_decoder_runner for token generation
  ET_LOG(Info, "Using decode method: %s", method_name.c_str());
  auto text_decoder_runner = std::make_unique<TextDecoderRunner>(
      module.get(), io_manager.get(), method_name);

  // Create optional prefill runner when a separate prefill method is specified
  std::unique_ptr<TextDecoderRunner> text_prefill_runner;
  TextDecoderRunner* prefill_runner_ptr = text_decoder_runner.get();
  if (!prefill_method_name.empty()) {
    ET_LOG(Info, "Using prefill method: %s", prefill_method_name.c_str());
    text_prefill_runner = std::make_unique<TextDecoderRunner>(
        module.get(), io_manager.get(), prefill_method_name);
    prefill_runner_ptr = text_prefill_runner.get();
  }

  // Create text_prefiller
  // Force parallel prefill when a separate prefill method is specified,
  // since the dedicated prefill method is exported for batch processing.
  bool enable_parallel_prefill =
      !prefill_method_name.empty() || metadata.at(kEnableDynamicShape);
  auto text_prefiller = std::make_unique<TextPrefiller>(
      prefill_runner_ptr,
      metadata.at(kUseKVCache),
      enable_parallel_prefill,
      metadata.at(kMaxSeqLen));

  // Create text_token_generator with stats
  auto stats = std::make_unique<Stats>();
  auto text_token_generator = std::make_unique<TextTokenGenerator>(
      tokenizer.get(),
      text_decoder_runner.get(),
      metadata.at(kUseKVCache),
      std::move(eos_ids),
      stats.get());

  // Create and return the Runner instance
  return std::make_unique<TextLLMRunner>(
      std::move(metadata),
      std::move(tokenizer),
      std::move(module),
      std::move(text_decoder_runner),
      std::move(text_prefiller),
      std::move(io_manager),
      std::move(text_token_generator),
      std::move(stats),
      temperature,
      std::move(text_prefill_runner),
      std::move(shared_method_memory));
}

std::unique_ptr<MultimodalRunner> create_multimodal_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::optional<const std::string> data_path,
    Module::LoadMode load_mode) {
  // Sanity check tokenizer
  if (!tokenizer || !tokenizer->is_loaded()) {
    ET_LOG(Error, "Tokenizer is null or not loaded");
    return nullptr;
  }

  // Create the Module
  std::unique_ptr<Module> module;
  if (data_path.has_value()) {
    module = std::make_unique<Module>(model_path, data_path.value(), load_mode);
  } else {
    module = std::make_unique<Module>(model_path, load_mode);
  }

  // Get metadata from Module
  ET_LOG(Info, "Reading metadata from model");
  auto metadata_result = get_llm_metadata(tokenizer.get(), module.get());
  if (metadata_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to get metadata from model");
    return nullptr;
  }
  auto metadata = metadata_result.get();

  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      get_eos_ids(tokenizer.get(), module.get()));

  // Create IOManager
  std::unique_ptr<IOManager> io_manager = std::make_unique<IOManager>(*module);

  // Create text_decoder_runner
  auto text_decoder_runner =
      std::make_unique<MultimodalDecoderRunner>(module.get(), io_manager.get());

  // Create multimodal_prefiller
  auto multimodal_prefiller = std::make_unique<MultimodalPrefiller>(
      module.get(),
      text_decoder_runner.get(),
      tokenizer.get(),
      io_manager.get());

  // Create text_token_generator with stats
  auto stats = std::make_unique<Stats>();
  auto text_token_generator = std::make_unique<TextTokenGenerator>(
      tokenizer.get(),
      text_decoder_runner.get(),
      metadata.at(kUseKVCache),
      std::move(eos_ids),
      stats.get());

  // Create and return the MultimodalRunner instance
  return std::make_unique<MultimodalRunner>(
      std::move(metadata),
      std::move(tokenizer),
      std::move(module),
      std::move(text_decoder_runner),
      std::move(multimodal_prefiller),
      std::move(io_manager),
      std::move(text_token_generator),
      std::move(stats));
}

} // namespace executorch::extension::llm
