/*
 * Mamba SSM model runner for ExecuTorch.
 *
 * Loads a Mamba .pte model and runs autoregressive generation.
 * Unlike transformer runners, Mamba uses a fixed-size SSM hidden state
 * instead of growing KV cache.
 *
 * Model IO:
 *   Input:  tokens(1, seq_len), ssm_states(n_layers, d_inner, d_state)
 *   Output: logits(1, seq_len, vocab_size), ssm_states(n_layers, d_inner, d_state)
 *
 * Usage:
 *   mamba_runner --model_path=mamba.pte --prompt="1,2,3" --max_tokens=20
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

static uint8_t method_allocator_pool[8 * 1024U * 1024U];  // 8 MB

DEFINE_string(model_path, "mamba.pte", "Path to the Mamba .pte model.");
DEFINE_string(prompt, "1,2,3", "Comma-separated token IDs as prompt.");
DEFINE_uint32(max_tokens, 20, "Maximum number of tokens to generate.");
DEFINE_uint32(n_layers, 4, "Number of SSM layers.");
DEFINE_uint32(d_inner, 128, "SSM inner dimension.");
DEFINE_uint32(d_state, 16, "SSM state dimension.");

using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::extension::FileDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorImpl;

namespace {

std::vector<int64_t> parse_prompt(const std::string& prompt_str) {
  std::vector<int64_t> tokens;
  std::stringstream ss(prompt_str);
  std::string item;
  while (std::getline(ss, item, ',')) {
    tokens.push_back(std::atol(item.c_str()));
  }
  return tokens;
}

int argmax(const float* data, size_t size) {
  return static_cast<int>(
      std::max_element(data, data + size) - data);
}

}  // namespace

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // --- Load program ---
  auto loader_result = FileDataLoader::from(FLAGS_model_path.c_str());
  ET_CHECK_MSG(loader_result.ok(), "Failed to open model: %s", FLAGS_model_path.c_str());
  auto loader = std::make_unique<FileDataLoader>(std::move(loader_result.get()));

  auto program_result = Program::load(loader.get());
  ET_CHECK_MSG(program_result.ok(), "Failed to load program.");
  auto program = std::move(program_result.get());

  auto method_name_result = program->get_method_name(0);
  ET_CHECK_MSG(method_name_result.ok(), "Failed to get method name.");
  const char* method_name = *method_name_result;
  ET_LOG(Info, "Method: %s", method_name);

  // --- Setup memory ---
  MemoryAllocator method_allocator(
      sizeof(method_allocator_pool), method_allocator_pool);

  auto method_meta_result = program->method_meta(method_name);
  ET_CHECK_MSG(method_meta_result.ok(), "Failed to get method meta.");
  auto& method_meta = method_meta_result.get();

  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers;
  std::vector<Span<uint8_t>> planned_spans;
  for (size_t i = 0; i < method_meta.num_memory_planned_buffers(); i++) {
    size_t buf_size = method_meta.memory_planned_buffer_size(i).get();
    planned_buffers.push_back(std::make_unique<uint8_t[]>(buf_size));
    planned_spans.push_back({planned_buffers.back().get(), buf_size});
  }

  HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});
  MemoryManager memory_manager(&method_allocator, &planned_memory);

  auto method_result = program->load_method(method_name, &memory_manager);
  ET_CHECK_MSG(method_result.ok(), "Failed to load method.");
  auto& method = method_result.get();

  // --- Parse prompt tokens ---
  auto prompt_tokens = parse_prompt(FLAGS_prompt);
  ET_LOG(Info, "Prompt tokens: %zu", prompt_tokens.size());

  // --- Allocate SSM state buffer (zeros) ---
  const size_t n_layers = FLAGS_n_layers;
  const size_t d_inner = FLAGS_d_inner;
  const size_t d_state = FLAGS_d_state;
  const size_t state_size = n_layers * d_inner * d_state;
  std::vector<float> ssm_state(state_size, 0.0f);

  // --- Prefill: process prompt tokens one at a time ---
  int64_t next_token = 0;
  SizesType token_shape[] = {1, 1};
  SizesType state_shape[] = {
      static_cast<SizesType>(n_layers),
      static_cast<SizesType>(d_inner),
      static_cast<SizesType>(d_state)};
  uint8_t token_dim_order[] = {0, 1};
  uint8_t state_dim_order[] = {0, 1, 2};

  ET_LOG(Info, "--- Prefill ---");
  for (size_t i = 0; i < prompt_tokens.size(); i++) {
    int64_t tok = prompt_tokens[i];

    TensorImpl token_impl(
        ScalarType::Long, 2, token_shape, &tok, token_dim_order);
    Tensor token_tensor(&token_impl);

    TensorImpl state_impl(
        ScalarType::Float, 3, state_shape, ssm_state.data(), state_dim_order);
    Tensor state_tensor(&state_impl);

    Error set_err = method.set_input(EValue(token_tensor), 0);
    ET_CHECK_MSG(set_err == Error::Ok, "Failed to set token input.");
    set_err = method.set_input(EValue(state_tensor), 1);
    ET_CHECK_MSG(set_err == Error::Ok, "Failed to set state input.");

    Error exec_err = method.execute();
    ET_CHECK_MSG(exec_err == Error::Ok, "Execution failed at prefill step %zu.", i);

    auto outputs = method.get_outputs();
    ET_CHECK_MSG(outputs.ok(), "Failed to get outputs.");

    // Update SSM state from output
    const auto& out_state = outputs.get()[1].toTensor();
    std::memcpy(
        ssm_state.data(),
        out_state.const_data_ptr<float>(),
        state_size * sizeof(float));

    // Get next token from logits
    const auto& logits = outputs.get()[0].toTensor();
    const float* logits_data = logits.const_data_ptr<float>();
    size_t vocab_size = logits.size(logits.dim() - 1);
    // Take logits from last position
    size_t last_pos_offset = (logits.numel() / vocab_size - 1) * vocab_size;
    next_token = argmax(logits_data + last_pos_offset, vocab_size);

    ET_LOG(Info, "Prefill [%zu]: tok=%ld -> next=%ld", i, (long)tok, (long)next_token);
  }

  // --- Decode: generate new tokens ---
  ET_LOG(Info, "--- Decode ---");
  std::vector<int64_t> generated_tokens;
  generated_tokens.push_back(next_token);

  for (uint32_t step = 0; step < FLAGS_max_tokens; step++) {
    int64_t tok = next_token;

    TensorImpl token_impl(
        ScalarType::Long, 2, token_shape, &tok, token_dim_order);
    Tensor token_tensor(&token_impl);

    TensorImpl state_impl(
        ScalarType::Float, 3, state_shape, ssm_state.data(), state_dim_order);
    Tensor state_tensor(&state_impl);

    Error set_err = method.set_input(EValue(token_tensor), 0);
    ET_CHECK_MSG(set_err == Error::Ok, "Failed to set token input.");
    set_err = method.set_input(EValue(state_tensor), 1);
    ET_CHECK_MSG(set_err == Error::Ok, "Failed to set state input.");

    Error exec_err = method.execute();
    ET_CHECK_MSG(exec_err == Error::Ok, "Execution failed at decode step %u.", step);

    auto outputs = method.get_outputs();
    ET_CHECK_MSG(outputs.ok(), "Failed to get outputs.");

    const auto& out_state = outputs.get()[1].toTensor();
    std::memcpy(
        ssm_state.data(),
        out_state.const_data_ptr<float>(),
        state_size * sizeof(float));

    const auto& logits = outputs.get()[0].toTensor();
    const float* logits_data = logits.const_data_ptr<float>();
    size_t vocab_size = logits.size(logits.dim() - 1);
    next_token = argmax(logits_data, vocab_size);

    generated_tokens.push_back(next_token);
    ET_LOG(Info, "Decode [%u]: tok=%ld -> next=%ld", step, (long)tok, (long)next_token);
  }

  // --- Print results ---
  std::printf("\nGenerated tokens: [");
  for (size_t i = 0; i < generated_tokens.size(); i++) {
    if (i > 0) std::printf(", ");
    std::printf("%ld", (long)generated_tokens[i]);
  }
  std::printf("]\n");

  return 0;
}
