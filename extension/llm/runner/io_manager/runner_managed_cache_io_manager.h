/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstring>

#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <executorch/extension/tensor/tensor.h>

namespace executorch {
namespace extension {
namespace llm {

/**
 * IOManager for models exported with runner_managed_cache=True.
 *
 * The model does NOT update KV cache internally. Instead:
 * - Prefill outputs: (logits, all_k, all_v)
 * - Decode inputs: (tokens, input_pos, k_cache, v_cache)
 * - Decode outputs: (logits, new_k, new_v)
 *
 * This IOManager allocates and maintains the KV cache, passes it as decode
 * input, and writes new K/V projections from model outputs into the cache.
 */
class ET_EXPERIMENTAL RunnerManagedCacheIOManager : public IOManager {
 public:
  RunnerManagedCacheIOManager(
      ET_MODULE_NAMESPACE::Module& module,
      int64_t n_layers,
      int64_t n_kv_heads,
      int64_t max_seq_len,
      int64_t head_dim,
      executorch::aten::ScalarType dtype)
      : IOManager(module),
        n_layers_(n_layers),
        n_kv_heads_(n_kv_heads),
        max_seq_len_(max_seq_len),
        head_dim_(head_dim),
        dtype_(dtype) {}

  ET_NODISCARD runtime::Error load(
      const std::string& prefill_method,
      const std::string& decode_method) override {
    (void)prefill_method;
    (void)decode_method;
    using SizesType = executorch::aten::SizesType;
    std::vector<SizesType> cache_shape = {
        static_cast<SizesType>(n_layers_),
        1,
        static_cast<SizesType>(n_kv_heads_),
        static_cast<SizesType>(max_seq_len_),
        static_cast<SizesType>(head_dim_)};
    k_cache_ = zeros(cache_shape, dtype_);
    v_cache_ = zeros(cache_shape, dtype_);
    return runtime::Error::Ok;
  }

  ET_NODISCARD runtime::Error reset(
      const std::string& prefill_method,
      const std::string& decode_method) override {
    (void)prefill_method;
    (void)decode_method;
    // Zero out the caches
    memset(k_cache_->mutable_data_ptr(), 0, k_cache_->nbytes());
    memset(v_cache_->mutable_data_ptr(), 0, v_cache_->nbytes());
    return runtime::Error::Ok;
  }

  runtime::Result<std::vector<runtime::EValue>> prepare_prefill(
      const TensorPtr& input,
      const TensorPtr& start_pos,
      const std::string& /*prefill_method*/) override {
    // Prefill takes (tokens, input_pos) — no cache input.
    last_start_pos_ = start_pos->const_data_ptr<int64_t>()[0];
    last_seq_len_ = input->numel();
    return std::vector<runtime::EValue>{input, start_pos};
  }

  runtime::Result<std::vector<runtime::EValue>> prepare_decode(
      const TensorPtr& input,
      const TensorPtr& start_pos,
      const std::string& decode_method) override {
    // TextDecoderRunner calls prepare_decode for ALL methods, including when
    // used for prefill. Detect the prefill method by checking its expected
    // number of inputs: prefill expects 2 (tokens, input_pos), decode expects
    // 4 (tokens, input_pos, k_cache, v_cache).
    auto method_meta = module().method_meta(decode_method);
    if (method_meta.ok() && method_meta->num_inputs() <= 2) {
      // This is actually the prefill method.
      return prepare_prefill(input, start_pos, decode_method);
    }
    // Decode takes (tokens, input_pos, k_cache, v_cache).
    last_start_pos_ = start_pos->const_data_ptr<int64_t>()[0];
    last_seq_len_ = 1;
    return std::vector<runtime::EValue>{input, start_pos, k_cache_, v_cache_};
  }

  ET_NODISCARD runtime::Error update_prefill(
      const std::vector<runtime::EValue>& model_outputs,
      const std::string& /*prefill_method*/) override {
    // model_outputs: [logits, all_k, all_v]
    // all_k shape: (n_layers, 1, n_kv_heads, seq_len, head_dim)
    if (model_outputs.size() < 3) {
      ET_LOG(Error, "Expected 3 outputs from prefill, got %zu",
             model_outputs.size());
      return runtime::Error::InvalidState;
    }
    const auto& all_k = model_outputs[1].toTensor();
    const auto& all_v = model_outputs[2].toTensor();
    return update_cache(all_k, all_v, last_start_pos_, last_seq_len_);
  }

  ET_NODISCARD runtime::Error update_decode(
      const std::vector<runtime::EValue>& model_outputs,
      const std::string& decode_method) override {
    // Both prefill and decode produce 3 outputs: [logits, k, v].
    // Detect prefill by seq_len stored during prepare.
    if (model_outputs.size() < 3) {
      ET_LOG(Error, "Expected 3 outputs, got %zu", model_outputs.size());
      return runtime::Error::InvalidState;
    }
    const auto& new_k = model_outputs[1].toTensor();
    const auto& new_v = model_outputs[2].toTensor();
    return update_cache(new_k, new_v, last_start_pos_, last_seq_len_);
  }

 private:
  runtime::Error update_cache(
      const executorch::aten::Tensor& src_k,
      const executorch::aten::Tensor& src_v,
      int64_t start_pos,
      int64_t seq_len) {
    // src_k shape: (n_layers, 1, n_kv_heads, seq_len, head_dim)
    // k_cache_ shape: (n_layers, 1, n_kv_heads, max_seq_len, head_dim)
    // Copy src into cache at positions start_pos..start_pos+seq_len-1.
    auto* dst_k = static_cast<uint8_t*>(k_cache_->mutable_data_ptr());
    auto* dst_v = static_cast<uint8_t*>(v_cache_->mutable_data_ptr());
    const auto* sk = static_cast<const uint8_t*>(src_k.const_data_ptr());
    const auto* sv = static_cast<const uint8_t*>(src_v.const_data_ptr());

    size_t elem_size = k_cache_->element_size();
    // Stride along the sequence dimension (dim 3) in cache.
    // Cache layout: (n_layers, 1, n_kv_heads, max_seq_len, head_dim)
    size_t head_stride = max_seq_len_ * head_dim_ * elem_size;
    size_t src_head_stride = seq_len * head_dim_ * elem_size;
    size_t copy_bytes = seq_len * head_dim_ * elem_size;
    size_t offset = start_pos * head_dim_ * elem_size;

    int64_t n_heads_total = n_layers_ * n_kv_heads_;  // batch_size=1
    for (int64_t h = 0; h < n_heads_total; ++h) {
      memcpy(dst_k + h * head_stride + offset, sk + h * src_head_stride,
             copy_bytes);
      memcpy(dst_v + h * head_stride + offset, sv + h * src_head_stride,
             copy_bytes);
    }
    return runtime::Error::Ok;
  }

  int64_t n_layers_;
  int64_t n_kv_heads_;
  int64_t max_seq_len_;
  int64_t head_dim_;
  executorch::aten::ScalarType dtype_;

  TensorPtr k_cache_;
  TensorPtr v_cache_;

  int64_t last_start_pos_{0};
  int64_t last_seq_len_{0};
};

} // namespace llm
} // namespace extension
} // namespace executorch
