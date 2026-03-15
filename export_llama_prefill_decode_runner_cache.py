#!/usr/bin/env python3
"""Export a Llama model with separate prefill and decode methods, runner-managed KV cache.

Exports two methods in a single .pte file:
  - kv_forward: accepts 1..max_seq_len tokens (dynamic shapes) for parallel prefill.
    Outputs (logits, new_k, new_v) where new_k/new_v have shape
    (n_layers, batch, n_kv_heads, seq_len, head_dim).
  - decode: accepts 1 token + external KV cache tensors.
    Outputs (logits, new_k, new_v) where new_k/new_v have shape
    (n_layers, batch, n_kv_heads, 1, head_dim).

The model does NOT update KV cache internally (no index_copy_ or slice assignment).
The C++ runner is responsible for maintaining the KV cache and writing new K/V
values at the correct positions after each forward call.
"""

import logging

import torch
from torch.export import Dim, export
from torch.nn.attention import SDPBackend
from torch.utils._pytree import LeafSpec

# Monkey-patch LeafSpec to fix a Python 3.10 + PyTorch version bug.
# LeafSpec is a frozen+slots dataclass with init=False fields that have
# defaults (type=None, _context=None, _children=[]). In Python 3.10,
# these defaults don't get stored on instances, so accessing .type fails.
# Fix by patching __post_init__ to explicitly set all fields.
_orig_leafspec_post_init = LeafSpec.__post_init__


def _leafspec_post_init(self):
    object.__setattr__(self, "type", None)
    object.__setattr__(self, "_context", None)
    object.__setattr__(self, "_children", [])
    object.__setattr__(self, "num_nodes", 1)
    object.__setattr__(self, "num_leaves", 1)
    object.__setattr__(self, "num_children", 0)


LeafSpec.__post_init__ = _leafspec_post_init

# Re-create the singleton with the fix applied.
import torch.utils._pytree as _pytree
import warnings as _warnings

with _warnings.catch_warnings():
    _warnings.filterwarnings(
        "ignore", category=FutureWarning, module=_pytree.__name__, append=False
    )
    _pytree._LEAF_SPEC = LeafSpec()

# Also fix deepcopy since all LeafSpecs are equivalent.
LeafSpec.__deepcopy__ = lambda self, memo: self
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


def _enable_runner_managed_cache(model):
    """Switch the model to runner-managed KV cache mode.

    In this mode the model does not write to internal cache buffers.
    Instead it returns new K/V projections for the runner to manage.
    """
    model.params.runner_managed_cache = True
    model.runner_managed_cache = True
    for layer in model.layers:
        if hasattr(layer, "attention"):
            attn = layer.attention
            attn.runner_managed_cache = True
            if hasattr(attn, "kv_cache"):
                del attn.kv_cache


def main():
    parser = build_args_parser()
    args = parser.parse_args()
    llm_config = LlmConfig.from_args(args)

    llm_config.model.use_kv_cache = True
    llm_config.model.enable_dynamic_shape = False

    max_seq_len = llm_config.export.max_seq_length

    builder = _prepare_for_llama_export(llm_config)
    model = builder.model

    _enable_runner_managed_cache(model)

    n_layers = model.params.n_layers
    n_kv_heads = model.params.n_kv_heads
    head_dim = model.params.head_dim
    dtype = next(model.parameters()).dtype

    # --- Export kv_forward (prefill) with dynamic sequence length ---
    # No external KV cache input; the model does self-attention only.
    # Returns (logits, all_k, all_v).
    # We call torch.export.export() directly instead of builder._export()
    # to skip run_decompositions() which has a LeafSpec deepcopy bug with
    # tuple outputs. Decomposition happens later in to_edge_transform_and_lower.
    logging.info("Exporting kv_forward with dynamic seq up to %d...", max_seq_len)
    seq_dim = Dim("seq", min=1, max=max_seq_len)
    prefill_args = (
        torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0),
        {"input_pos": torch.arange(0, max_seq_len, dtype=torch.long)},
    )
    prefill_dynamic_shapes = (
        {1: seq_dim},
        {"input_pos": {0: seq_dim}},
    )
    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        kv_forward_prog = export(
            model,
            prefill_args,
            dynamic_shapes=prefill_dynamic_shapes,
            strict=True,
        )

    # --- Export decode with external KV cache ---
    # Runner passes k_cache and v_cache as inputs.
    # Returns (logits, new_k, new_v).
    logging.info("Exporting decode with external KV cache...")
    k_cache = torch.zeros(
        n_layers, 1, n_kv_heads, max_seq_len, head_dim, dtype=dtype
    )
    v_cache = torch.zeros_like(k_cache)
    decode_args = (
        torch.tensor([[1]], dtype=torch.long),
        {
            "input_pos": torch.tensor([0], dtype=torch.long),
            "k_cache": k_cache,
            "v_cache": v_cache,
        },
    )
    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        decode_prog = export(
            model,
            decode_args,
            dynamic_shapes=None,
            strict=True,
        )

    # Metadata for the C++ runner.
    builder.metadata["enable_dynamic_shape"] = True
    builder.metadata["runner_managed_cache"] = True

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
        output_file = output_file[:-4] + "_prefill_decode_rc.pte"

    save_pte_program(export_program, output_file, builder.output_dir)
    logging.info(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
