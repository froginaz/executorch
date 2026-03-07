/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/samsung/runtime/enn_executor.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>

DEFINE_string(
    model_path,
    "llama.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(tokenizer_path, "tokenizer.model", "Tokenizer file path.");

DEFINE_string(prompt, "The answer to the ultimate question is", "Prompt.");

DEFINE_double(
    temperature,
    0.8f,
    "Temperature; 0 = greedy argmax sampling (deterministic).");

DEFINE_int32(
    seq_len,
    128,
    "Total number of tokens to generate (prompt + output).");

DEFINE_int32(
    max_new_tokens,
    -1,
    "Number of new tokens to generate, excluding the prompt.");

DEFINE_bool(warmup, false, "Whether to run a warmup run.");

DEFINE_bool(
    ignore_eos,
    false,
    "Whether to ignore EOS token and continue generating.");

DEFINE_string(
    method_name,
    "forward",
    "Method name to execute in the model.");

DEFINE_string(
    prefill_method_name,
    "",
    "Optional separate method name for the prefill phase.");

using namespace torch::executor::enn;

struct EnnApiDeinit {
  void operator()(EnnApi* ptr) const {
    if (ptr == nullptr || ptr->EnnDeinitialize == nullptr) {
      return;
    }
    auto ret = ptr->EnnDeinitialize();
    ET_CHECK_MSG(ret == ENN_RET_SUCCESS, "Enn Deinitialize failed.");
  }
};

std::unique_ptr<EnnApi, EnnApiDeinit> exynos_npu_init() {
  EnnApi* enn_api_inst = EnnApi::getEnnApiInstance();
  if (enn_api_inst->EnnInitialize == nullptr) {
    ET_LOG(Info, "ENN API library not available. Skipping NPU initialization.");
    return std::unique_ptr<EnnApi, EnnApiDeinit>(nullptr);
  }
  auto ret = enn_api_inst->EnnInitialize();
  ET_CHECK_MSG(ret == ENN_RET_SUCCESS, "Enn initialize failed.");
  return std::unique_ptr<EnnApi, EnnApiDeinit>(enn_api_inst);
}

int main(int argc, char** argv) {
  std::unique_ptr<EnnApi, EnnApiDeinit> instance = exynos_npu_init();

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto tokenizer =
      executorch::extension::llm::load_tokenizer(FLAGS_tokenizer_path);
  if (tokenizer == nullptr) {
    ET_LOG(Error, "Failed to load tokenizer from %s", FLAGS_tokenizer_path.c_str());
    return 1;
  }

  auto runner = executorch::extension::llm::create_text_llm_runner(
      FLAGS_model_path,
      std::move(tokenizer),
      /*data_path=*/std::nullopt,
      FLAGS_temperature,
      FLAGS_method_name,
      executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors,
      FLAGS_prefill_method_name);

  if (runner == nullptr) {
    ET_LOG(Error, "Failed to create llama runner");
    return 1;
  }

  if (FLAGS_warmup) {
    int32_t warmup_max_new_tokens =
        FLAGS_max_new_tokens != -1 ? FLAGS_max_new_tokens : FLAGS_seq_len;
    auto error = runner->warmup(
        FLAGS_prompt.c_str(), /*max_new_tokens=*/warmup_max_new_tokens);
    if (error != executorch::runtime::Error::Ok) {
      ET_LOG(Error, "Failed to warmup llama runner");
      return 1;
    }
  }

  executorch::extension::llm::GenerationConfig config{
      .temperature = static_cast<float>(FLAGS_temperature)};

  config.ignore_eos = FLAGS_ignore_eos;

  if (FLAGS_max_new_tokens != -1) {
    config.max_new_tokens = FLAGS_max_new_tokens;
  } else {
    config.seq_len = FLAGS_seq_len;
  }

  auto error = runner->generate(FLAGS_prompt.c_str(), config);
  if (error != executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to run llama runner");
    return 1;
  }

  return 0;
}
