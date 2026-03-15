# Runner-Managed KV Cache (No ATen Cache Ops in the Graph)

Export a Llama model where the KV cache is **entirely managed by the C++ runner**,
not by ATen ops inside the model graph. The exported model contains no
`index_copy_`, slice assignment, or any in-place cache mutation ops. Instead,
each forward call returns new K/V projections, and the runner writes them into
the cache at the correct positions.

## Why runner-managed cache?

### The standard approach: model-managed cache

In the default ExecuTorch Llama export, the model graph contains KV cache
buffers as mutable state. During inference, ATen ops like `index_copy_` or
custom SDPA ops update these buffers in-place:

```
Model graph contains:
  k_cache = self.kv_cache.k_cache          # registered buffer
  k_cache.index_copy_(2, input_pos, new_k) # ATen op mutates buffer in-place
  v_cache.index_copy_(2, input_pos, new_v)
  output = sdpa(q, k_cache, v_cache, ...)
  return logits
```

This works, but the cache update logic is baked into the exported graph. The
graph includes mutation ops that some backends may not support or cannot optimize.
With two methods (prefill + decode), both methods must share the same physical
memory via `share_mutable_buffers=True` in memory planning.

### The runner-managed approach

The model graph has **no cache mutation ops at all**. Each forward call is a pure
function that returns new K/V projections alongside logits:

```
Model graph contains:
  new_k, new_v = linear_proj(x)            # fresh K/V from this step
  k_full = scatter(external_cache_k, pos, new_k)  # out-of-place combine (decode only)
  output = sdpa(q, k_full, v_full, ...)
  return logits, new_k, new_v              # runner handles the rest
```

The C++ runner:
1. Allocates the KV cache as a plain tensor
2. Passes it as an input to the decode method
3. Receives new K/V outputs and writes them into the cache via `memcpy`

### Comparison

| Aspect | Model-Managed | Runner-Managed |
|--------|---------------|----------------|
| Cache mutation ops in graph | `index_copy_`, slice assign | None |
| Model output | `(logits,)` | `(logits, new_k, new_v)` |
| Decode inputs | `(tokens, input_pos)` | `(tokens, input_pos, k_cache, v_cache)` |
| Cross-method memory sharing | `share_mutable_buffers=True` | Not needed |
| Backend compatibility | Requires mutable buffer support | Any backend (pure function) |
| Cache layout control | Fixed by export | Runner can choose layout |

## Architecture

### Two exported methods

The .pte file contains two methods:

**`kv_forward`** (prefill) — processes 1..N prompt tokens in parallel:
```
Inputs:  tokens (1, seq_len), input_pos (seq_len,)
Outputs: logits (1, vocab), all_k (n_layers, 1, n_kv_heads, seq_len, head_dim),
                            all_v (n_layers, 1, n_kv_heads, seq_len, head_dim)
```
No external cache input. The model does self-attention over the input tokens
only, using a causal mask `mask[input_pos][:, input_pos]` that covers just the
prefill range. Returns all K/V projections from every layer, stacked into a
single tensor.

**`decode`** — processes 1 token with the full KV context:
```
Inputs:  tokens (1, 1), input_pos (1,), k_cache (n_layers, 1, n_kv_heads, max_seq, head_dim),
                                        v_cache (same shape)
Outputs: logits (1, vocab), new_k (n_layers, 1, n_kv_heads, 1, head_dim),
                            new_v (same shape)
```
Receives the full KV cache from the runner. Each attention layer extracts its
slice (`k_cache[layer_id]`), uses `scatter` to insert the new K/V at `input_pos`,
then runs SDPA over the combined context. Returns per-layer K/V projections for
the runner to append.

### Data flow

```
                        ┌──────────────────┐
  Prompt tokens ──────> │   kv_forward     │ ──> logits ──> first token
                        │   (prefill)      │ ──> all_k (n_layers, 1, n_kv, seq, hd)
                        └──────────────────┘ ──> all_v
                                                   │
                                    memcpy into cache at pos [0..seq-1]
                                                   │
                                                   v
                        ┌──────────────────┐   ┌─────────┐
  Next token ─────────> │     decode       │ <─│ k_cache │ (runner passes as input)
  input_pos=[seq] ────> │                  │ <─│ v_cache │
                        └──────────────────┘   └─────────┘
                           │         │
                           v         v
                        logits    new_k, new_v (single position)
                                     │
                          memcpy into cache at pos [seq]
                                     │
                                  (repeat)
```

## How it works in the model

Three files are modified to support `runner_managed_cache`:

### 1. ModelArgs flag (`model_args.py`)

```python
@dataclass
class ModelArgs:
    use_kv_cache: bool = False
    runner_managed_cache: bool = False  # NEW: KV cache updated by runner, not model
```

### 2. Attention layer (`attention.py`)

When `runner_managed_cache=True`:

- **No `KVCache` module is created** — the internal cache buffers are skipped entirely
- **Prefill path** (no external cache): Self-attention over input tokens only,
  with mask `self.mask[input_pos][:, input_pos]` (a submatrix of the causal mask)
- **Decode path** (external cache provided): Uses `torch.scatter` (out-of-place)
  to insert new K/V at `input_pos`, then SDPA over the full combined context
- **Returns `(output, {"new_k": k, "new_v": v})`** instead of just `output`

Key detail on the mask indexing — `self.mask[input_pos][:, input_pos]`:
- For prefill: selects a `(seq, seq)` causal submatrix that only covers the
  input range. This avoids `self.mask[:seqlen, :seqlen]` which causes
  `torch.export` to specialize when `seqlen == max_seq_len`.
- For decode with cache: `self.mask[input_pos]` gives a `(1, max_seq)` row,
  which correctly masks all cached positions plus the current one.

Key detail on `scatter` vs `index_copy_`:
- `index_copy_` is in-place (mutates the cache tensor) — this is what
  model-managed cache uses, and it shows up as an ATen op in the graph
- `scatter` is out-of-place (returns a new tensor) — no mutation, clean graph,
  any backend can handle it

```python
# In AttentionMHA.forward(), before the existing KV cache path:
if self.runner_managed_cache:
    new_k, new_v = k, v
    k_cache_ext = kwargs.get("k_cache")
    if k_cache_ext is not None:
        # Decode: merge new K/V into external cache (out-of-place)
        layer_k = k_cache_ext[self.layer_id]
        layer_v = kwargs["v_cache"][self.layer_id]
        indices = input_pos.reshape(1, 1, -1, 1).expand_as(k)
        k = layer_k.scatter(2, indices, k)  # out-of-place — no graph mutation
        v = layer_v.scatter(2, indices, v)
        attn_mask = self.mask[input_pos]
    else:
        # Prefill: self-attention only (no external cache)
        attn_mask = self.mask[input_pos][:, input_pos]
    output = self.SDPA(input_pos, q, k, v, bsz, seqlen, attn_mask)
    return self.wo(output), {"new_k": new_k, "new_v": new_v}
```

### 3. Transformer (`llama_transformer.py`)

Collects per-layer K/V outputs and stacks them:

```python
new_kvs_k = []
new_kvs_v = []
for layer in self.layers:
    h, attn_options_update = layer(h, freqs_cos, freqs_sin, attn_options_)
    if self.runner_managed_cache:
        if attn_options_update is not None and "new_k" in attn_options_update:
            new_kvs_k.append(attn_options_update["new_k"])
            new_kvs_v.append(attn_options_update["new_v"])
    elif attn_options_update is not None:
        attn_options_.update(**attn_options_update)

# After norm + output projection:
if self.runner_managed_cache and new_kvs_k:
    all_k = torch.stack(new_kvs_k)  # (n_layers, batch, n_kv_heads, seq, head_dim)
    all_v = torch.stack(new_kvs_v)
    return logits, all_k, all_v
```

## Export

### Export script

`export_llama_prefill_decode_runner_cache.py` in the repo root.

```bash
python export_llama_prefill_decode_runner_cache.py \
  -c ~/.llama/checkpoints/Llama3.2-1B/consolidated.00.pth \
  -p ~/.llama/checkpoints/Llama3.2-1B/params.json \
  --max_seq_length 128 -X -kv
```

Output: `llama3_2_prefill_decode_rc.pte`

### What the export script does

1. **Prepare model** via `_prepare_for_llama_export()` (loads checkpoint, applies quantization)
2. **Enable runner-managed cache** — sets `runner_managed_cache=True` on model,
   params, and all attention layers; deletes any existing `KVCache` modules
3. **Export `kv_forward`** with dynamic shapes `Dim("seq", min=1, max=max_seq_len)`
4. **Export `decode`** with fixed shapes, including `k_cache` and `v_cache` as inputs
5. **Set metadata**: `runner_managed_cache=True`, `enable_dynamic_shape=True`
6. **Lower both methods** via `to_edge_transform_and_lower` with XNNPACK partitioners
7. **Memory planning** with `alloc_graph_input=False` (cache inputs are managed externally)
   — NO `share_mutable_buffers` needed since there are no mutable buffers to share

### Key differences from model-managed export

| | Model-Managed (`export_llama_prefill_decode.py`) | Runner-Managed (`..._runner_cache.py`) |
|---|---|---|
| `_enable_runner_managed_cache()` | Not called | Called — deletes KVCache modules |
| Decode inputs | `(tokens, input_pos)` | `(tokens, input_pos, k_cache, v_cache)` |
| `share_mutable_buffers` | `True` (critical) | Not needed |
| `alloc_graph_input` | Default | `False` (runner allocates cache) |
| Metadata | `use_kv_cache=True` | `runner_managed_cache=True` |
| Export call | `builder._export()` | `torch.export.export()` directly |

The export calls `torch.export.export()` directly instead of `builder._export()`
to skip `run_decompositions()` which has a `LeafSpec` deepcopy bug with tuple
outputs in certain PyTorch versions. Decomposition happens later in
`to_edge_transform_and_lower`.

## C++ Runner

### RunnerManagedCacheIOManager

`extension/llm/runner/io_manager/runner_managed_cache_io_manager.h`

This `IOManager` subclass handles all cache management:

**`load()`**: Allocates `k_cache_` and `v_cache_` tensors with shape
`(n_layers, 1, n_kv_heads, max_seq_len, head_dim)`.

**`prepare_prefill()`**: Returns `{tokens, input_pos}` — no cache input.
Stores `last_seq_len_ = num_tokens` for the subsequent `update_decode()`.

**`prepare_decode()`**: Detects prefill vs decode by checking the method's
`num_inputs()` (2 = prefill, 4 = decode). For decode, returns
`{tokens, input_pos, k_cache_, v_cache_}`.

**`update_decode()`**: Extracts `outputs[1]` (new_k) and `outputs[2]` (new_v),
copies them into the cache at `last_start_pos_` via memcpy:

```
For each (layer, head):
  memcpy(cache + head_stride * h + start_pos * head_dim * elem_size,
         src + src_head_stride * h,
         seq_len * head_dim * elem_size)
```

### Automatic detection

The runner auto-detects runner-managed cache via metadata. In
`llm_runner_helper.cpp`, when `create_text_llm_runner()` is called:

1. Reads `runner_managed_cache` from model metadata (stored as a constant method)
2. If true, reads the decode method's input metadata to get `k_cache` shape
   (input index 2) → extracts `n_layers`, `n_kv_heads`, `max_seq_len`, `head_dim`, `dtype`
3. Creates `RunnerManagedCacheIOManager` with these dimensions
4. Skips shared mutable buffer setup (not needed)

### Build requirement

`llm_runner_helper.cpp` must be included in the CMake build. It is listed in
`build/executorch_srcs.cmake` under `_extension_llm_runner__srcs`:

```cmake
set(_extension_llm_runner__srcs
    extension/llm/runner/llm_runner_helper.cpp   # must be present
    extension/llm/runner/text_decoder_runner.cpp
    extension/llm/runner/text_llm_runner.cpp
    extension/llm/runner/text_prefiller.cpp
    extension/llm/sampler/sampler.cpp
)
```

Build and run:

```bash
make llama-cpu

./cmake-out/examples/models/llama/llama_main \
  --model_path llama3_2_prefill_decode_rc.pte \
  --tokenizer_path ~/.llama/checkpoints/Llama3.2-1B/tokenizer.model \
  --prompt "The sky is" \
  --seq_len 128 \
  --temperature 0 \
  --method_name decode \
  --prefill_method_name kv_forward
```

## Files changed

| File | Change |
|------|--------|
| `examples/models/llama/model_args.py` | Add `runner_managed_cache: bool` to `ModelArgs` |
| `examples/models/llama/attention.py` | Runner-managed cache path in `AttentionMHA`: skip `KVCache` creation, `scatter`-based decode, return `new_k`/`new_v` |
| `examples/models/llama/llama_transformer.py` | Collect per-layer K/V, return `(logits, all_k, all_v)` |
| `export_llama_prefill_decode_runner_cache.py` | New export script |
| `extension/llm/runner/constants.h` | Add `kRunnerManagedCache` constant |
| `extension/llm/runner/io_manager/io_manager.h` | Add `protected module()` accessor |
| `extension/llm/runner/io_manager/runner_managed_cache_io_manager.h` | New IOManager subclass |
| `extension/llm/runner/llm_runner_helper.cpp` | Auto-detect and create `RunnerManagedCacheIOManager` |
| `extension/llm/runner/text_decoder_runner.cpp` | Allow >= 1 outputs (was exactly 1) |

## Troubleshooting

### `LeafSpec` error during export
```
AttributeError: 'LeafSpec' object has no attribute 'type'
```
Python 3.10 + PyTorch frozen dataclass bug. The export script includes a
monkey-patch for `LeafSpec.__post_init__` that fixes this. If you hit it
elsewhere, ensure the export script's patch runs before any `torch.export` call.

### `ConstraintViolationError: seq specialized to N`
The causal mask indexing `self.mask[:seqlen, :seqlen]` causes torch.export to
specialize when `seqlen == max_seq_len`. The fix (already applied) uses
`self.mask[input_pos][:, input_pos]` which is tensor-indexed and avoids specialization.

### Decode produces garbage or EOS immediately
- Verify the runner binary includes `RunnerManagedCacheIOManager`. Check:
  `nm llama_main | grep RunnerManaged` — should show symbols.
- Verify `build/executorch_srcs.cmake` includes `llm_runner_helper.cpp`.
- Rebuild with `make llama-cpu` after changes.

### No generated tokens (only prompt echoed)
The runner binary may be stale and lack the runner-managed cache code path.
Rebuild after ensuring `llm_runner_helper.cpp` is in the build sources.
