# `enable_dynamic_shape` in ExecuTorch LLM Export

`enable_dynamic_shape` appears in three layers — model config, export builder,
and runtime metadata — each controlling different behavior.

## Layer 1: Model config (`ModelArgs.enable_dynamic_shape`)

Controls **which code paths** the Llama model uses internally during tracing.

### Rope (rotary position embedding) — `rope.py:294-309`

```python
if self.params.enable_dynamic_shape:
    # Uses .item() → FAILS with torch.export dynamic shapes
    input_pos_item = input_pos[-1].item()
    torch._check_is_size(input_pos_item)
    freqs_cos = self.freqs_cos.narrow(0, input_pos_item, seq_len)
else:
    # Direct indexing → works with torch.export
    freqs_cos = self.freqs_cos[input_pos]
```

### KV cache update — `attention.py:92-108`

```python
if self.enable_dynamic_shape:
    # Uses .item() + index_copy_
    start_pos = input_pos[0].item()
    self.k_cache.index_copy_(dim_to_slice, indices, k_val)
else:
    # Direct indexing
    k_out[:, :, input_pos] = k_val
```

### Attention position — `attention.py:479-484`

```python
if self.enable_dynamic_shape:
    start_pos = input_pos[-1].item()  # scalar extraction
    seq_length = q.size(2)
```

**Key point**: When `True`, the model calls `.item()` which creates unbacked
SymInts during `torch.export` tracing. This causes
`RuntimeError: Cannot cast FakeTensor to number`. When `False`, direct tensor
indexing works cleanly with both static and dynamic `torch.export` shapes.

### Example inputs — `model.py:286-302`

```python
def get_example_inputs_kvcache_sdpa(self):
    if self.enable_dynamic_shape:
        return (
            torch.tensor([[2, 3, 4]], dtype=torch.long),     # multi-token
            {"input_pos": torch.tensor([0], dtype=torch.long)},
        )
    else:
        return (
            torch.tensor([[1]], dtype=torch.long),            # single token
            {"input_pos": torch.tensor([0], dtype=torch.long)},
        )
```

## Layer 2: Export builder (`LLMEdgeManager.enable_dynamic_shape`)

Controls whether the **export builder auto-generates** `dynamic_shapes` for
`torch.export.export()`.

From `builder.py:139-152`:
```python
if not self.dynamic_shapes and self.enable_dynamic_shape:
    if not self.use_kv_cache:
        self.dynamic_shapes = (
            {1: torch.export.Dim("token_dim", max=self.max_seq_len - 1)},
        )
    else:
        self.dynamic_shapes = (
            {1: torch.export.Dim("token_dim", max=self.max_seq_len - 1)},
            {"input_pos": {0: 1}},  # input_pos stays static at size 1
        )
```

When `True`, token dimension 1 becomes dynamic (`[1..max_seq_len-1]`).
When `False`, all dimensions are static (no `dynamic_shapes` generated).

This auto-generation can be overridden by setting `builder.dynamic_shapes`
manually before calling `builder._export()`.

## Layer 3: Runtime metadata (`enable_dynamic_shape` in .pte)

Stored as a constant method in the .pte file. The C++ runner reads it to
decide **prefill strategy**.

### Parallel vs sequential prefill — `llm_runner_helper.cpp:332-333`

```cpp
bool enable_parallel_prefill =
    !prefill_method_name.empty() || metadata.at(kEnableDynamicShape);
```

| Metadata value | Prefill behavior |
|---|---|
| `True` | **Parallel** — all prompt tokens at once as `{1, N}` |
| `False` | **Sequential** — tokens one-by-one as `{1, 1}` |

### input_pos tensor shape — `util.h:111-150`

```cpp
auto numel = second_input_sizes[0];  // from method metadata
if (numel > 1) {
    // Range: [start_pos, start_pos+1, ..., start_pos+N-1]
} else {
    // Scalar: [start_pos]
}
```

The method's `input_tensor_meta(1).sizes()[0]` determines the format. For a
method exported with dynamic shapes, this is the upper bound (e.g., 128).

## Backend compatibility

Not all backends support dynamic shapes:

```python
# export_llama_lib.py:861-866
if llm_config.model.enable_dynamic_shape and (
    llm_config.backend.coreml.enabled
    or llm_config.backend.mps.enabled
    or llm_config.backend.qnn.enabled
):
    raise ValueError(...)
```

| Backend | Dynamic shape support |
|---------|---------------------|
| XNNPACK | Yes |
| Vulkan | Yes (via `require_dynamic_shapes` config) |
| CoreML | No (use multifunction approach instead) |
| MPS | No |
| QNN | No |

## The two-method trick

For the prefill/decode split export (`export_llama_prefill_decode.py`), the
model config and metadata are set **independently**:

```python
llm_config.model.enable_dynamic_shape = False   # Model internals: simple rope/cache paths
                                                  # (avoids .item() failure in torch.export)

builder.metadata["enable_dynamic_shape"] = True  # Runtime metadata: tells C++ runner
                                                  # to use parallel prefill
```

The `dynamic_shapes` parameter to `torch.export.export()` (from `torch.export.Dim`)
is a **separate mechanism** that controls which dimensions the export framework
treats as variable. It does not depend on or require the model's internal
`enable_dynamic_shape` flag.

## CLI flag

```bash
# Default: enabled
python -m executorch.examples.models.llama.export_llama_lib ...

# Disable:
python -m executorch.examples.models.llama.export_llama_lib --disable_dynamic_shape
```

Note the inverted name: `--disable_dynamic_shape` sets
`enable_dynamic_shape=False`.

## Summary

| Layer | Flag | Default | Controls |
|-------|------|---------|----------|
| Model config | `ModelArgs.enable_dynamic_shape` | `False` | `.item()` vs direct indexing in rope/cache/attention |
| Export builder | `LLMEdgeManager.enable_dynamic_shape` | `True` | Auto-generation of `torch.export` `dynamic_shapes` |
| Runtime metadata | `enable_dynamic_shape()` in .pte | from config | Parallel vs sequential prefill in C++ runner |
