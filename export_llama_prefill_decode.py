#!/usr/bin/env python3
"""Export a Llama model with separate kv_forward (prefill) and decode methods.

Exports two methods in a single .pte file:
  - kv_forward: accepts 1..max_seq_len tokens (dynamic shapes) for parallel prefill
  - decode: accepts exactly 1 token for autoregressive generation

Both methods share the same KV cache buffers via share_mutable_buffers=True in the
memory planning pass, so state written by kv_forward is visible to decode at runtime.

Uses enable_dynamic_shape=False in the model (simpler rope path) while passing
dynamic_shapes to torch.export for kv_forward's runtime flexibility.
"""

import logging

import torch
from torch.export import Dim
from executorch.backends.xnnpack._passes.convert_to_linear import ConvertToLinearPass
from executorch.examples.models.llama.export_llama_lib import (
    _get_output_filename,
    _get_xnnpack_partitioners,
    _prepare_for_llama_export,
    build_args_parser,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.extension.export_util.utils import save_pte_program
from executorch.extension.llm.export.config.llm_config import LlmConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = build_args_parser()
    args = parser.parse_args()
    llm_config = LlmConfig.from_args(args)

    llm_config.model.use_kv_cache = True
    llm_config.model.enable_dynamic_shape = False

    max_seq_len = llm_config.export.max_seq_length

    builder = _prepare_for_llama_export(llm_config)

    # --- Export kv_forward with dynamic sequence length (1..max_seq_len) ---
    logging.info("Exporting kv_forward with dynamic seq up to %d...", max_seq_len)
    seq_dim = Dim("seq", min=1, max=max_seq_len)
    builder.example_inputs = (
        torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0),
        {"input_pos": torch.arange(0, max_seq_len, dtype=torch.long)},
    )
    builder.dynamic_shapes = (
        {1: seq_dim},
        {"input_pos": {0: seq_dim}},
    )
    kv_forward_prog = builder._export()

    # --- Export decode with fixed single-token shape ---
    logging.info("Exporting decode with seq_len=1...")
    builder.example_inputs = (
        torch.tensor([[1]], dtype=torch.long),
        {"input_pos": torch.tensor([0], dtype=torch.long)},
    )
    builder.dynamic_shapes = None
    decode_prog = builder._export()

    # enable_dynamic_shape=True tells the C++ runner to use parallel prefill.
    builder.metadata["enable_dynamic_shape"] = True

    # --- Lower both methods together ---
    logging.info("Lowering...")
    partitioners = _get_xnnpack_partitioners(llm_config)
    edge_config = builder._get_edge_config()

    edge_manager = to_edge_transform_and_lower(
        {"kv_forward": kv_forward_prog, "decode": decode_prog},
        partitioner=partitioners,
        compile_config=edge_config,
        constant_methods=builder.metadata,
    )

    logging.info("Converting to ExecuTorch program...")
    edge_manager.transform([ConvertToLinearPass()])
    export_program = edge_manager.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            passes=[],
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                share_mutable_buffers=True,
            ),
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )

    output_file = _get_output_filename(
        llm_config,
        builder.modelname,
        builder.output_dir,
        builder.dtype,
    )
    if output_file.endswith(".pte"):
        output_file = output_file[:-4] + "_prefill_decode.pte"

    save_pte_program(export_program, output_file, builder.output_dir)
    logging.info(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
