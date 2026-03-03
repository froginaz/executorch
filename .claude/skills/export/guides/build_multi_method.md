# Multi-Method .pte Export (Prefill/Decode Split)

Export two methods (`kv_forward` for parallel prefill, `decode` for single-token
generation) into a single `.pte` file with shared KV cache.

## Why two methods?

A single-method model with fixed `seq_len=1` must prefill prompt tokens
sequentially (one-by-one). A dedicated prefill method accepts variable-length
input and processes all prompt tokens in parallel — typically 2-4x faster.

## Quick recipe

Three changes turn a single-method export into a two-method export:

```python
from torch.export import Dim, export
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass

# 1. Export the SAME model twice with different input shapes
seq_dim = Dim("seq", min=1, max=max_seq_len)

kv_forward_prog = export(model, prefill_inputs, dynamic_shapes=prefill_dynamic_shapes)
decode_prog     = export(model, decode_inputs)   # fixed seq_len=1, no dynamic_shapes

# 2. Pass as Dict[str, ExportedProgram] (not a single program)
edge = to_edge_transform_and_lower(
    {"kv_forward": kv_forward_prog, "decode": decode_prog},
    partitioner=partitioners,
    compile_config=edge_config,
    constant_methods=metadata,
)

# 3. Enable shared mutable buffers
et = edge.to_executorch(ExecutorchBackendConfig(
    memory_planning_pass=MemoryPlanningPass(share_mutable_buffers=True),
))
```

## How each change works

### Change 1: Two `torch.export.export()` calls

The model is exported twice from the **same `nn.Module` instance**. Both calls
use FakeTensors for tracing — the model's actual state is not modified. Both
`ExportedProgram`s reference the same registered buffers (KV cache) by their
fully-qualified names.

**kv_forward** — dynamic shapes for variable-length prefill:
```python
seq_dim = Dim("seq", min=1, max=max_seq_len)

builder.example_inputs = (
    torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0),  # {1, max_seq_len}
    {"input_pos": torch.arange(0, max_seq_len, dtype=torch.long)},     # {max_seq_len}
)
builder.dynamic_shapes = (
    {1: seq_dim},                     # tokens dim 1: dynamic [1..max_seq_len]
    {"input_pos": {0: seq_dim}},      # input_pos dim 0: dynamic [1..max_seq_len]
)
kv_forward_prog = builder._export()
```

**decode** — fixed single-token shape:
```python
builder.example_inputs = (
    torch.tensor([[1]], dtype=torch.long),              # {1, 1}
    {"input_pos": torch.tensor([0], dtype=torch.long)}, # {1}
)
builder.dynamic_shapes = None
decode_prog = builder._export()
```

> **Note on `enable_dynamic_shape`**: The model config uses
> `enable_dynamic_shape=False` (simpler rope path using `self.freqs_cos[input_pos]`
> indexing). The `dynamic_shapes` parameter to `torch.export` is separate — it
> tells the export framework which dimensions are variable, without affecting
> model internals. Setting `enable_dynamic_shape=True` in the model config
> triggers a `.item()` call in rope.py that fails with `torch.export`.

### Change 2: Dict input to `to_edge_transform_and_lower`

```python
# Single method:
to_edge_transform_and_lower(program, ...)          # → 1 execution plan "forward"

# Multiple methods:
to_edge_transform_and_lower(
    {"kv_forward": kv_forward_prog, "decode": decode_prog},  # → 2 execution plans
    ...
)
```

When given a `Dict[str, ExportedProgram]`:
- Each entry becomes a separate execution plan (method) in the .pte
- Constant data (weights) is **deduplicated** across methods in the flatbuffer
- Each method gets its own operator graph, delegate partitions, and I/O specs
- Mutable buffers are tracked by fully-qualified name across all methods

### Change 3: `share_mutable_buffers=True`

This is the critical flag that enables KV cache sharing.

```python
MemoryPlanningPass(share_mutable_buffers=True)
```

**Without** this flag: each method's mutable buffers (KV cache) are planned
independently. Even with the same allocator at runtime, the offsets within the
buffer differ — kv_forward writes to offset X, decode reads from offset Y.
Result: decode sees uninitialized memory → garbage output.

**With** this flag: the memory planner runs a cross-method coordination step
(`run_multimethod()`) after planning each method individually:

1. Collects all mutable buffer TensorSpecs that share the same FQN across methods
   (e.g., `layers.0.attention.kv_cache.k_cache` appears in both kv_forward and decode)
2. Moves them from `mem_id=1` (activations) to `mem_id=2` (shared state)
3. Assigns **identical `mem_offset`** to each FQN across all methods

Result in the .pte:
```
kv_forward: non_const_buffer_sizes = [0, 17MB,  8MB]
decode:     non_const_buffer_sizes = [0,  1MB,  8MB]
                                      │    │     └─ mem_id=2: shared KV cache (SAME size + offsets)
                                      │    └─ mem_id=1: activations (method-specific)
                                      └─ reserved
```

## Runtime: How the C++ runner uses two methods

### Shared memory setup

When `--prefill_method_name` is set, `create_text_llm_runner()` in
`llm_runner_helper.cpp` performs:

1. **Query both methods' buffer requirements** via `MethodMeta`:
   ```
   kv_forward: 3 buffers [0, 17MB, 8MB]
   decode:     3 buffers [0,  1MB, 8MB]
   ```

2. **Allocate shared buffers** — for each buffer index, allocate `max(kv_forward, decode)`:
   ```
   buffer[0] = 0 bytes    (reserved)
   buffer[1] = 17 MB      (max of activations — safe because only one method runs at a time)
   buffer[2] = 8 MB       (shared KV cache — identical in both)
   ```

3. **Create one `HierarchicalAllocator`** from these buffers

4. **Load both methods with the same allocator**:
   ```cpp
   module->load_method("kv_forward", shared_allocator);
   module->load_method("decode",     shared_allocator);
   ```

Both methods resolve their KV cache tensors via `buffer[2].data() + mem_offset`.
Since the offsets are identical (ensured by `share_mutable_buffers=True`), both
methods read/write the exact same physical memory addresses.

### Execution flow

```
generate("The sky is blue")
  │
  ├─ Tokenize → [791, 13180, 374, 6437]
  │
  ├─ PREFILL via kv_forward:
  │   TextPrefiller → TextDecoderRunner(method_name="kv_forward")
  │     tokens: {1, 4}  input_pos: [0,1,2,3]
  │     → Writes KV cache positions 0-3 in buffer[2]
  │     → Returns logits → next token
  │
  └─ DECODE via decode (token-by-token loop):
      TextTokenGenerator → TextDecoderRunner(method_name="decode")
        tokens: {1, 1}  input_pos: [4]
        → Reads KV cache positions 0-3 from buffer[2]  ← written by kv_forward!
        → Writes position 4
        → Returns logits → next token
        (repeat with pos=5,6,7...)
```

### `populate_start_pos_or_cache_position` behavior

This utility reads `method_meta.input_tensor_meta(1).sizes()[0]` to determine
the `input_pos` format:

| Method | `input_tensor_meta(1).sizes()[0]` | `input_pos` format |
|--------|----------------------------------|-------------------|
| kv_forward | 128 (dynamic upper bound) | Range: `[start_pos, start_pos+1, ..., start_pos+N-1]` |
| decode | 1 (fixed) | Scalar: `[start_pos]` |

## Complete Llama example

See `export_llama_prefill_decode.py` in the repo root:

```bash
# Export
python export_llama_prefill_decode.py \
  -c ~/.llama/checkpoints/Llama3.2-1B/consolidated.00.pth \
  -p ~/.llama/checkpoints/Llama3.2-1B/params.json \
  --max_seq_length 128 --use_sdpa_with_kv_cache -X -kv

# Run
./cmake-out/examples/models/llama/llama_main \
  --model_path llama3_prefill_decode.pte \
  --tokenizer_path ~/.llama/checkpoints/Llama3.2-1B/tokenizer.model \
  --prompt "What is the capital of France?" \
  --method_name decode \
  --prefill_method_name kv_forward
```

## Troubleshooting

### Garbage output from decode after prefill
KV cache not shared. Verify `share_mutable_buffers=True` was used during export.
Check: both execution plans should have 3 entries in `non_const_buffer_sizes`
and `mem_id=2` values should have identical offsets.

### rope `.item()` error during export
```
RuntimeError: Cannot cast FakeTensor to number
```
The model's `enable_dynamic_shape=True` triggers `.item()` in `rope.py` which
is incompatible with `torch.export`. Use `enable_dynamic_shape=False` in the
model config and pass `dynamic_shapes` to `torch.export` instead.

### `ConstraintViolationError` on dynamic shapes
The `Dim` range may need adjustment. Ensure `min=1` (not 0, to avoid 0/1
specialization issues). Use explicit `Dim("seq", min=1, max=max_seq_len)`.

### Large .pte file size
With XNNPACK delegation, delegate segments may duplicate weight data. The
two-method .pte can be ~2x the single-method size. Weights in the constant
buffer are deduplicated, but delegate-internal copies are not.
