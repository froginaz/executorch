/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */

/**
 * @file
 *
 * This tool can run ExecuTorch model files that contain separate prefill and
 * decode methods sharing a KV-cache via externally allocated buffers.
 * On Android it uses the ENN backend; on x86 it uses XNNPACK.
 */

#ifdef __ANDROID__
#include <executorch/backends/samsung/runtime/enn_executor.h>
#include <executorch/backends/samsung/runtime/profile.hpp>
#else
#define EXYNOS_ATRACE_BEGIN(name)
#define EXYNOS_ATRACE_END()
#endif
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>
#include <gflags/gflags.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>

static uint8_t method_allocator_pool[8 * 1024U * 1024U]; // 8 MB (two methods)
static uint8_t prefill_allocator_pool[4 * 1024U * 1024U]; // 4 MB

DEFINE_string(model, "model.pte", "Model serialized in flatbuffer format.");
DEFINE_string(
    prefill_method,
    "prefill",
    "Name of the prefill method in the PTE.");
DEFINE_string(
    decode_method,
    "decode",
    "Name of the decode method in the PTE.");
DEFINE_string(
    input,
    "",
    "Input file path, support multiple inputs: input_1 input_2 ...");
DEFINE_uint32(num_executions, 1, "Number of decode steps to run.");
DEFINE_int32(warm_up, 0, "Pre-run before inference.");
DEFINE_bool(dump_statistics, false, "Dump inference statistics.");
DEFINE_string(output_path, "", "Output Execution results to target directory.");

using namespace torch::executor;
using torch::executor::util::FileDataLoader;
#ifdef __ANDROID__
using namespace torch::executor::enn;
#endif

std::vector<std::string> split(std::string str, char delimiter = ' ') {
  std::vector<std::string> result;
  std::stringstream ss(str);
  std::string temp;
  while (std::getline(ss, temp, delimiter)) {
    if (!temp.empty()) {
      result.push_back(temp);
    }
  }
  return result;
}

class DataReader {
 public:
  typedef std::vector<uint8_t> data_t;

  DataReader(size_t size) : data_set_(size) {}

  void read(const std::string file_path) {
    ET_CHECK(index_ < data_set_.size());
    data_t& data = data_set_[index_];
    std::ifstream input_file(file_path.c_str(), std::ios::binary);
    ET_CHECK(input_file.is_open());
    input_file.seekg(0, std::ios::end);
    data.resize(input_file.tellg());
    input_file.seekg(0);
    input_file.read(reinterpret_cast<char*>(data.data()), data.size());
    input_file.close();
    ++index_;
  }

  void alloc(const int size) {
    ET_CHECK(index_ < data_set_.size());
    data_t& data = data_set_[index_];
    data.resize(size);
    ++index_;
  }

  void* get(int32_t index) {
    ET_CHECK(index < data_set_.size());
    return data_set_[index].data();
  }

  size_t nbytes(int32_t index) {
    ET_CHECK(index < data_set_.size());
    return data_set_[index].size();
  }

  ~DataReader() = default;

 private:
  std::vector<data_t> data_set_;
  int32_t index_ = 0;
};

void saveOutput(const exec_aten::Tensor& tensor, int32_t output_index) {
  if (FLAGS_output_path.empty()) {
    return;
  }
  auto output_file_name =
      FLAGS_output_path + "/output_" + std::to_string(output_index) + ".bin";
  std::ofstream fout(output_file_name.c_str(), std::ios::binary);
  ET_CHECK_MSG(
      fout.is_open(),
      "Directory or have no visit permission: %s",
      FLAGS_output_path.c_str());
  fout.write(tensor.const_data_ptr<char>(), tensor.nbytes());
  fout.close();
}

#ifdef __ANDROID__
struct EnnApiDeinit {
  void operator()(EnnApi* ptr) const {
    if (ptr == nullptr) {
      return;
    }

    auto ret = ptr->EnnDeinitialize();
    ET_CHECK_MSG(ret == ENN_RET_SUCCESS, "Enn Deinitialize failed.");
  }
};

std::unique_ptr<EnnApi, EnnApiDeinit> exynos_npu_init() {
  EnnApi* enn_api_inst = EnnApi::getEnnApiInstance();
  auto ret = enn_api_inst->EnnInitialize();
  ET_CHECK_MSG(ret == ENN_RET_SUCCESS, "Enn initialize failed.");
  return std::unique_ptr<EnnApi, EnnApiDeinit>(enn_api_inst);
}
#endif

struct SharedPlannedMemory {
  std::vector<std::vector<uint8_t>> buffers;
  std::vector<Span<uint8_t>> spans;
  std::unique_ptr<HierarchicalAllocator> allocator;
};

SharedPlannedMemory create_shared_planned_memory(
    const MethodMeta& prefill_meta,
    const MethodMeta& decode_meta) {
  SharedPlannedMemory shared;

  size_t prefill_bufs = prefill_meta.num_memory_planned_buffers();
  size_t decode_bufs = decode_meta.num_memory_planned_buffers();
  size_t num_buffers = std::max(prefill_bufs, decode_bufs);

  shared.buffers.reserve(num_buffers);
  shared.spans.reserve(num_buffers);

  for (size_t i = 0; i < num_buffers; ++i) {
    int64_t pf_size =
        (i < prefill_bufs)
        ? prefill_meta.memory_planned_buffer_size(i).get()
        : 0;
    int64_t dc_size =
        (i < decode_bufs)
        ? decode_meta.memory_planned_buffer_size(i).get()
        : 0;
    size_t max_size = static_cast<size_t>(std::max(pf_size, dc_size));

    ET_LOG(
        Info,
        "Shared planned buffer %zu: prefill=%lld, decode=%lld, alloc=%zu",
        i,
        (long long)pf_size,
        (long long)dc_size,
        max_size);

    shared.buffers.emplace_back(max_size);
    shared.spans.emplace_back(
        shared.buffers.back().data(), max_size);
  }

  shared.allocator = std::make_unique<HierarchicalAllocator>(Span<Span<uint8_t>>(
      shared.spans.data(), shared.spans.size()));

  return shared;
}

int main(int argc, char** argv) {
  auto before_init = std::chrono::high_resolution_clock::now();
#ifdef __ANDROID__
  std::unique_ptr<EnnApi, EnnApiDeinit> instance = exynos_npu_init();
#endif
  auto after_init = std::chrono::high_resolution_clock::now();
  double interval_init = std::chrono::duration_cast<std::chrono::microseconds>(
                             after_init - before_init)
                             .count() /
      1000.0;

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 1) {
    std::string msg = "Extra commandline args:";
    for (int i = 1 /* skip argv[0] (program name) */; i < argc; i++) {
      msg += std::string(" ") + argv[i];
    }
    ET_LOG(Error, "%s", msg.c_str());
    return 1;
  }

  const char* model_path = FLAGS_model.c_str();
  Result<FileDataLoader> loader = FileDataLoader::from(model_path);
  ET_CHECK_MSG(
      loader.ok(),
      "FileDataLoader::from() failed: 0x%" PRIx32,
      (uint32_t)loader.error());

  Result<Program> program = Program::load(&loader.get());
  if (!program.ok()) {
    ET_LOG(Error, "Failed to parse model file %s", model_path);
    return 1;
  }
  ET_LOG(Info, "Model file %s is loaded.", model_path);

  const char* prefill_name = FLAGS_prefill_method.c_str();
  const char* decode_name = FLAGS_decode_method.c_str();

  Result<MethodMeta> prefill_meta = program->method_meta(prefill_name);
  ET_CHECK_MSG(
      prefill_meta.ok(),
      "Failed to get method_meta for %s: 0x%" PRIx32,
      prefill_name,
      (uint32_t)prefill_meta.error());

  Result<MethodMeta> decode_meta = program->method_meta(decode_name);
  ET_CHECK_MSG(
      decode_meta.ok(),
      "Failed to get method_meta for %s: 0x%" PRIx32,
      decode_name,
      (uint32_t)decode_meta.error());

  // Allocate shared planned memory (includes KV-cache on mem_id=2).
  // Both methods get the same physical buffers so KV-cache state written
  // by prefill is visible to decode without any copy.
  auto shared_memory =
      create_shared_planned_memory(*prefill_meta, *decode_meta);

  // Separate method allocators for each method's internal metadata.
  MemoryAllocator prefill_method_allocator{MemoryAllocator(
      sizeof(prefill_allocator_pool), prefill_allocator_pool)};
  MemoryAllocator decode_method_allocator{MemoryAllocator(
      sizeof(method_allocator_pool), method_allocator_pool)};

  MemoryManager prefill_memory_manager(
      &prefill_method_allocator, shared_memory.allocator.get());
  MemoryManager decode_memory_manager(
      &decode_method_allocator, shared_memory.allocator.get());

  // Load prefill method
  EXYNOS_ATRACE_BEGIN("Load prefill method");
  auto before_load = std::chrono::high_resolution_clock::now();
  Result<Method> prefill_method =
      program->load_method(prefill_name, &prefill_memory_manager);
  ET_CHECK_MSG(
      prefill_method.ok(),
      "Loading prefill method %s failed with status 0x%" PRIx32,
      prefill_name,
      (uint32_t)prefill_method.error());
  ET_LOG(Info, "Prefill method %s loaded.", prefill_name);
  EXYNOS_ATRACE_END();

  // Load decode method (shares planned memory with prefill)
  EXYNOS_ATRACE_BEGIN("Load decode method");
  Result<Method> decode_method =
      program->load_method(decode_name, &decode_memory_manager);
  auto after_load = std::chrono::high_resolution_clock::now();
  double interval_load = std::chrono::duration_cast<std::chrono::microseconds>(
                             after_load - before_load)
                             .count() /
      1000.0;
  ET_CHECK_MSG(
      decode_method.ok(),
      "Loading decode method %s failed with status 0x%" PRIx32,
      decode_name,
      (uint32_t)decode_method.error());
  ET_LOG(Info, "Decode method %s loaded.", decode_name);
  EXYNOS_ATRACE_END();

  // --- Prefill phase ---
  ET_LOG(Info, "Running prefill...");
  bool _is_input_arg_existed = (FLAGS_input != "");
  auto input_files = split(FLAGS_input);
  DataReader prefill_data_reader(prefill_method->inputs_size());

  EXYNOS_ATRACE_BEGIN("Prepare prefill input");
  if (!_is_input_arg_existed) {
    auto inputs = executorch::extension::prepare_input_tensors(*prefill_method);
    ET_CHECK_MSG(
        inputs.ok(),
        "Could not prepare prefill inputs: 0x%" PRIx32,
        (uint32_t)inputs.error());
    ET_LOG(Info, "Prefill inputs prepared with default values.");
  } else {
    ET_CHECK_MSG(
        input_files.size() == prefill_method->inputs_size(),
        "Please check the number of given input binary files");
    for (const auto& input_file : input_files) {
      prefill_data_reader.read(input_file);
    }
    for (int input_index = 0; input_index < prefill_method->inputs_size();
         ++input_index) {
      MethodMeta meta = prefill_method->method_meta();
      Result<TensorInfo> tensor_meta = meta.input_tensor_meta(input_index);
      ET_CHECK_MSG(
          prefill_data_reader.nbytes(input_index) == tensor_meta->nbytes(),
          "Given prefill input size is invalid");
      TensorImpl impl = TensorImpl(
          tensor_meta->scalar_type(),
          tensor_meta->sizes().size(),
          const_cast<TensorImpl::SizesType*>(tensor_meta->sizes().data()),
          prefill_data_reader.get(input_index),
          const_cast<TensorImpl::DimOrderType*>(
              tensor_meta->dim_order().data()));
      Error ret = prefill_method->set_input(Tensor(&impl), input_index);
      ET_CHECK_MSG(
          ret == Error::Ok, "Failed to set prefill input tensor: %d", ret);
    }
  }
  EXYNOS_ATRACE_END();

  auto before_prefill = std::chrono::high_resolution_clock::now();
  Error status = prefill_method->execute();
  auto after_prefill = std::chrono::high_resolution_clock::now();
  double interval_prefill =
      std::chrono::duration_cast<std::chrono::microseconds>(
          after_prefill - before_prefill)
          .count() /
      1000.0;
  ET_CHECK_MSG(
      status == Error::Ok,
      "Prefill execution failed with status 0x%" PRIx32,
      static_cast<int32_t>(status));
  ET_LOG(Info, "Prefill done in %f ms.", interval_prefill);

  // --- Decode phase ---
  // KV-cache is already populated by prefill in the shared planned memory.
  ET_LOG(Info, "Running %d decode steps...", FLAGS_num_executions);

  // Warm up
  for (int i = 0; i < FLAGS_warm_up; ++i) {
    auto inputs = executorch::extension::prepare_input_tensors(*decode_method);
    ET_CHECK_MSG(inputs.ok(), "Could not prepare decode warm-up inputs");
    decode_method->execute();
  }

  auto before_decode = std::chrono::high_resolution_clock::now();
  for (uint32_t i = 0; i < FLAGS_num_executions; ++i) {
    auto inputs = executorch::extension::prepare_input_tensors(*decode_method);
    ET_CHECK_MSG(inputs.ok(), "Could not prepare decode inputs");
    status = decode_method->execute();
    ET_CHECK_MSG(
        status == Error::Ok,
        "Decode step %d failed with status 0x%" PRIx32,
        i,
        static_cast<int32_t>(status));
  }
  auto after_decode = std::chrono::high_resolution_clock::now();
  double interval_decode =
      std::chrono::duration_cast<std::chrono::microseconds>(
          after_decode - before_decode)
          .count() /
      1000.0;

  ET_LOG(
      Info,
      "%d decode steps took %f ms, avg %f ms/step",
      FLAGS_num_executions,
      interval_decode,
      interval_decode / (float)FLAGS_num_executions);

  if (FLAGS_dump_statistics) {
    auto output_file_name = "statistics.txt";
    std::ofstream fout(output_file_name);
    fout << "init: " + std::to_string(interval_init)
         << "\nload: " + std::to_string(interval_load)
         << "\nprefill: " + std::to_string(interval_prefill)
         << "\ndecode_total: " + std::to_string(interval_decode)
         << "\ndecode_avg: " +
            std::to_string(interval_decode / (float)FLAGS_num_executions)
         << std::endl;
    fout.close();
  }

  // Get outputs from decode method
  std::vector<EValue> outputs(decode_method->outputs_size());
  status = decode_method->get_outputs(outputs.data(), outputs.size());
  ET_CHECK(status == Error::Ok);

  for (size_t output_index = 0; output_index < decode_method->outputs_size();
       ++output_index) {
    auto output_tensor = outputs[output_index].toTensor();
    saveOutput(output_tensor, output_index);
  }

  return 0;
}
