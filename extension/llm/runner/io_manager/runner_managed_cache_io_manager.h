/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstring>
#include <vector>

#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>

namespace executorch {
namespace extension {
namespace llm {

/**
 * IO manager for models where the runner manages KV-cache externally.
 *
 * The model receives past KV-cache as input and produces new KV-cache entries
 * as output. The runner is responsible for concatenating outputs into the
 * cache and feeding the updated cache back on the next step.
 *
 * Supports micro-batch prefill: the runner may call prepare_prefill /
 * update_prefill multiple times (each processing prefill_seq_len tokens)
 * before switching to decode. The KV-cache accumulates in the prefill
 * buffer across micro-batches and is copied to the decode buffer on the
 * first prepare_decode call.
 *
 * Prefill (per micro-batch):
 *   Input:  token[prefill_seq_len], start_pos, attn_mask[R,C],
 *           k_cache[1,H,past,D] x L, v_cache[1,H,past,D] x L
 *   Output: logits, k_cache[1,H,prefill_seq_len,D] x L,
 *           v_cache[1,H,prefill_seq_len,D] x L
 *   where past = max_seq_len - prefill_seq_len
 *
 * Decode:
 *   Input:  token[1], start_pos, attn_mask[R,C],
 *           k_cache[1,H,max-1,D] x L, v_cache[1,H,max-1,D] x L
 *   Output: logits, k_cache[1,H,1,D] x L, v_cache[1,H,1,D] x L
 */
class RunnerManagedCacheIOManager : public IOManager {
 public:
  struct Config {
    size_t max_seq_len;
    size_t prefill_seq_len;
    size_t n_layers;
    size_t n_kv_heads;
    size_t head_dim;
    size_t attn_mask_rows = 128;
    size_t attn_mask_cols = 1024;
  };

  RunnerManagedCacheIOManager(
      ET_MODULE_NAMESPACE::Module& module,
      Config config)
      : IOManager(module), config_(config) {
    prefill_cache_len_ = config_.max_seq_len - config_.prefill_seq_len;
    decode_cache_len_ = config_.max_seq_len - 1;
    head_size_ = config_.n_kv_heads * config_.head_dim;

    attn_mask_.resize(
        config_.attn_mask_rows * config_.attn_mask_cols, 0.0f);
    allocate_buffers();
  }

  ET_NODISCARD runtime::Error load(
      const std::string& prefill_method,
      const std::string& decode_method) override {
    (void)prefill_method;
    (void)decode_method;
    return runtime::Error::Ok;
  }

  ET_NODISCARD runtime::Error reset(
      const std::string& prefill_method,
      const std::string& decode_method) override {
    (void)prefill_method;
    (void)decode_method;
    current_pos_ = 0;
    prefill_done_ = false;
    for (size_t l = 0; l < config_.n_layers; l++) {
      std::fill(
          prefill_k_input_[l].begin(), prefill_k_input_[l].end(), 0.0f);
      std::fill(
          prefill_v_input_[l].begin(), prefill_v_input_[l].end(), 0.0f);
      std::fill(
          decode_k_input_[l].begin(), decode_k_input_[l].end(), 0.0f);
      std::fill(
          decode_v_input_[l].begin(), decode_v_input_[l].end(), 0.0f);
    }
    std::fill(attn_mask_.begin(), attn_mask_.end(), 0.0f);
    return runtime::Error::Ok;
  }

  runtime::Result<std::vector<runtime::EValue>> prepare_prefill(
      const TensorPtr& input,
      const TensorPtr& start_pos,
      const std::string& prefill_method) override {
    (void)prefill_method;

    update_attn_mask(current_pos_, config_.prefill_seq_len);

    std::vector<runtime::EValue> inputs;
    inputs.reserve(3 + config_.n_layers * 2);
    inputs.emplace_back(input);
    inputs.emplace_back(start_pos);

    attn_mask_tensor_ = from_blob(
        attn_mask_.data(),
        {static_cast<executorch::aten::SizesType>(config_.attn_mask_rows),
         static_cast<executorch::aten::SizesType>(config_.attn_mask_cols)});
    inputs.emplace_back(attn_mask_tensor_);

    prefill_k_input_tensors_.clear();
    prefill_v_input_tensors_.clear();
    prefill_k_output_tensors_.clear();
    prefill_v_output_tensors_.clear();

    for (size_t l = 0; l < config_.n_layers; l++) {
      auto k_in = from_blob(
          prefill_k_input_[l].data(),
          {1,
           static_cast<executorch::aten::SizesType>(config_.n_kv_heads),
           static_cast<executorch::aten::SizesType>(prefill_cache_len_),
           static_cast<executorch::aten::SizesType>(config_.head_dim)});
      auto v_in = from_blob(
          prefill_v_input_[l].data(),
          {1,
           static_cast<executorch::aten::SizesType>(config_.n_kv_heads),
           static_cast<executorch::aten::SizesType>(prefill_cache_len_),
           static_cast<executorch::aten::SizesType>(config_.head_dim)});
      prefill_k_input_tensors_.push_back(k_in);
      prefill_v_input_tensors_.push_back(v_in);
      inputs.emplace_back(k_in);
      inputs.emplace_back(v_in);
    }

    return inputs;
  }

  runtime::Result<std::vector<runtime::EValue>> prepare_decode(
      const TensorPtr& input,
      const TensorPtr& start_pos,
      const std::string& decode_method) override {
    (void)decode_method;

    if (!prefill_done_) {
      copy_prefill_to_decode_input();
      prefill_done_ = true;
    }

    update_attn_mask(current_pos_, 1);

    std::vector<runtime::EValue> inputs;
    inputs.reserve(3 + config_.n_layers * 2);
    inputs.emplace_back(input);
    inputs.emplace_back(start_pos);

    attn_mask_tensor_ = from_blob(
        attn_mask_.data(),
        {1,
         static_cast<executorch::aten::SizesType>(config_.attn_mask_cols)});
    inputs.emplace_back(attn_mask_tensor_);

    decode_k_input_tensors_.clear();
    decode_v_input_tensors_.clear();
    decode_k_output_tensors_.clear();
    decode_v_output_tensors_.clear();

    for (size_t l = 0; l < config_.n_layers; l++) {
      auto k_in = from_blob(
          decode_k_input_[l].data(),
          {1,
           static_cast<executorch::aten::SizesType>(config_.n_kv_heads),
           static_cast<executorch::aten::SizesType>(decode_cache_len_),
           static_cast<executorch::aten::SizesType>(config_.head_dim)});
      auto v_in = from_blob(
          decode_v_input_[l].data(),
          {1,
           static_cast<executorch::aten::SizesType>(config_.n_kv_heads),
           static_cast<executorch::aten::SizesType>(decode_cache_len_),
           static_cast<executorch::aten::SizesType>(config_.head_dim)});
      decode_k_input_tensors_.push_back(k_in);
      decode_v_input_tensors_.push_back(v_in);
      inputs.emplace_back(k_in);
      inputs.emplace_back(v_in);
    }

    return inputs;
  }

  ET_NODISCARD runtime::Error update_prefill(
      const std::vector<runtime::EValue>& model_outputs,
      const std::string& prefill_method) override {
    (void)prefill_method;
    // model_outputs: [logits, k0, v0, k1, v1, ..., k15, v15]
    // k_out shape: [1, n_kv_heads, prefill_seq_len, head_dim]
    // Accumulate prefill output into prefill input buffers. The copy to
    // decode input is deferred until the first prepare_decode call so that
    // multiple micro-batch prefill iterations work correctly.

    for (size_t l = 0; l < config_.n_layers; l++) {
      const auto& k_out = model_outputs[1 + l * 2].toTensor();
      const auto& v_out = model_outputs[1 + l * 2 + 1].toTensor();
      const float* k_data = k_out.const_data_ptr<float>();
      const float* v_data = v_out.const_data_ptr<float>();

      copy_to_cache(
          k_data,
          config_.prefill_seq_len,
          prefill_k_input_[l].data(),
          prefill_cache_len_,
          current_pos_);
      copy_to_cache(
          v_data,
          config_.prefill_seq_len,
          prefill_v_input_[l].data(),
          prefill_cache_len_,
          current_pos_);
    }
    current_pos_ += config_.prefill_seq_len;
    return runtime::Error::Ok;
  }

  ET_NODISCARD runtime::Error update_decode(
      const std::vector<runtime::EValue>& model_outputs,
      const std::string& decode_method) override {
    (void)decode_method;
    // model_outputs: [logits, k0, v0, k1, v1, ..., k15, v15]
    // k_out shape: [1, n_kv_heads, 1, head_dim]
    // Copy decode output into decode input cache at current_pos_.

    for (size_t l = 0; l < config_.n_layers; l++) {
      const auto& k_out = model_outputs[1 + l * 2].toTensor();
      const auto& v_out = model_outputs[1 + l * 2 + 1].toTensor();
      const float* k_data = k_out.const_data_ptr<float>();
      const float* v_data = v_out.const_data_ptr<float>();

      copy_to_cache(
          k_data,
          1,
          decode_k_input_[l].data(),
          decode_cache_len_,
          current_pos_);
      copy_to_cache(
          v_data,
          1,
          decode_v_input_[l].data(),
          decode_cache_len_,
          current_pos_);
    }
    current_pos_ += 1;
    return runtime::Error::Ok;
  }

  size_t current_pos() const {
    return current_pos_;
  }

 private:
  void allocate_buffers() {
    size_t prefill_in_size = config_.n_kv_heads * prefill_cache_len_ *
        config_.head_dim;
    size_t decode_in_size = config_.n_kv_heads * decode_cache_len_ *
        config_.head_dim;

    prefill_k_input_.resize(config_.n_layers);
    prefill_v_input_.resize(config_.n_layers);
    decode_k_input_.resize(config_.n_layers);
    decode_v_input_.resize(config_.n_layers);

    for (size_t l = 0; l < config_.n_layers; l++) {
      prefill_k_input_[l].resize(prefill_in_size, 0.0f);
      prefill_v_input_[l].resize(prefill_in_size, 0.0f);
      decode_k_input_[l].resize(decode_in_size, 0.0f);
      decode_v_input_[l].resize(decode_in_size, 0.0f);
    }
  }

  /**
   * Copy prefill output KV cache (layout [1,H,prefill_cache_len,D]) into
   * the decode input buffer (layout [1,H,decode_cache_len,D]).
   *
   * Since the head stride differs between the two layouts, we copy per-head.
   */
  void copy_prefill_to_decode_input() {
    for (size_t l = 0; l < config_.n_layers; l++) {
      for (size_t h = 0; h < config_.n_kv_heads; h++) {
        const float* src = prefill_k_input_[l].data() +
            h * prefill_cache_len_ * config_.head_dim;
        float* dst = decode_k_input_[l].data() +
            h * decode_cache_len_ * config_.head_dim;
        std::memcpy(
            dst, src, prefill_cache_len_ * config_.head_dim * sizeof(float));
      }
      for (size_t h = 0; h < config_.n_kv_heads; h++) {
        const float* src = prefill_v_input_[l].data() +
            h * prefill_cache_len_ * config_.head_dim;
        float* dst = decode_v_input_[l].data() +
            h * decode_cache_len_ * config_.head_dim;
        std::memcpy(
            dst, src, prefill_cache_len_ * config_.head_dim * sizeof(float));
      }
    }
  }

  /**
   * Copy new cache entries from output (layout [1,H,new_len,D]) into the
   * decode input buffer (layout [1,H,cache_len,D]) at the given position.
   *
   * For each head h:
   *   cache[h * cache_len * D + pos * D .. + (pos + new_len) * D]
   *     = src[h * new_len * D .. + new_len * D]
   */
  void copy_to_cache(
      const float* src,
      size_t new_len,
      float* cache,
      size_t cache_len,
      size_t pos) {
    for (size_t h = 0; h < config_.n_kv_heads; h++) {
      const float* src_head = src + h * new_len * config_.head_dim;
      float* dst_head =
          cache + h * cache_len * config_.head_dim + pos * config_.head_dim;
      std::memcpy(
          dst_head, src_head, new_len * config_.head_dim * sizeof(float));
    }
  }

  /**
   * Update the attention mask for a causal (lower-triangular) pattern.
   * For each query row r in [0, seq_len), the mask allows attending to
   * positions [0, pos + r] (value 1) and masks future positions (value 0).
   */
  void update_attn_mask(size_t pos, size_t seq_len) {
    std::fill(attn_mask_.begin(), attn_mask_.end(), 0.0f);
    for (size_t r = 0; r < seq_len && r < config_.attn_mask_rows; r++) {
      size_t visible = pos + r + 1;
      if (visible > config_.attn_mask_cols) {
        visible = config_.attn_mask_cols;
      }
      for (size_t c = 0; c < visible; c++) {
        attn_mask_[r * config_.attn_mask_cols + c] = 1.0f;
      }
    }
  }

  Config config_;
  size_t prefill_cache_len_;
  size_t decode_cache_len_;
  size_t head_size_;
  size_t current_pos_ = 0;
  bool prefill_done_ = false;

  // Per-layer input buffers: [1, n_kv_heads, cache_len, head_dim] flattened
  std::vector<std::vector<float>> prefill_k_input_;
  std::vector<std::vector<float>> prefill_v_input_;
  std::vector<std::vector<float>> decode_k_input_;
  std::vector<std::vector<float>> decode_v_input_;

  // Attention mask buffer: [attn_mask_rows, attn_mask_cols] flattened
  std::vector<float> attn_mask_;
  TensorPtr attn_mask_tensor_;

  // Tensor wrappers (kept alive between prepare and model execution)
  std::vector<TensorPtr> prefill_k_input_tensors_;
  std::vector<TensorPtr> prefill_v_input_tensors_;
  std::vector<TensorPtr> prefill_k_output_tensors_;
  std::vector<TensorPtr> prefill_v_output_tensors_;
  std::vector<TensorPtr> decode_k_input_tensors_;
  std::vector<TensorPtr> decode_v_input_tensors_;
  std::vector<TensorPtr> decode_k_output_tensors_;
  std::vector<TensorPtr> decode_v_output_tensors_;
};

} // namespace llm
} // namespace extension
} // namespace executorch
