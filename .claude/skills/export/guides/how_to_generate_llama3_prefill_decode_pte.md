# How to Generate `llama3_prefill_decode.pte`

End-to-end walkthrough: from Llama checkpoint to a two-method .pte with shared
KV cache, then running it with `llama_main`.

## Prerequisites

- ExecuTorch installed (`./install_executorch.sh` or `pip install -e .`)
- Llama 3.2 1B checkpoint downloaded to `~/.llama/checkpoints/Llama3.2-1B/`
  - `consolidated.00.pth` — model weights
  - `params.json` — model config
  - `tokenizer.model` — SentencePiece tokenizer
- C++ runner built (`./build_llama_runner.sh`)

## Step 1: Export

```bash
python export_llama_prefill_decode.py \
  -c ~/.llama/checkpoints/Llama3.2-1B/consolidated.00.pth \
  -p ~/.llama/checkpoints/Llama3.2-1B/params.json \
  --max_seq_length 128 \
  --use_sdpa_with_kv_cache \
  -X -kv
```

### What each flag does

| Flag | Effect |
|------|--------|
| `-c` | Path to the checkpoint `.pth` file |
| `-p` | Path to the `params.json` config |
| `--max_seq_length 128` | Upper bound for dynamic sequence dimension |
| `--use_sdpa_with_kv_cache` | Use the fused SDPA+KV-cache op (faster attention) |
| `-X` | Delegate eligible ops to the XNNPACK backend |
| `-kv` | Enable KV cache (required for prefill/decode split) |

Output: `llama3_2_prefill_decode.pte` (~5.6 GB for Llama 3.2 1B with XNNPACK).

### What the script does internally

```
1. _prepare_for_llama_export(llm_config)
   → Loads checkpoint, builds LlamaTransformer, applies quantization
   → Sets enable_dynamic_shape=False (avoids .item() in rope)
   → Returns a LLMEdgeManager (builder)

2. Export kv_forward (prefill method)
   → example_inputs: tokens={1, 128}, input_pos={128}
   → dynamic_shapes: both dims dynamic [1..128]
   → builder._export() → torch.export.export()

3. Export decode (generation method)
   → example_inputs: tokens={1, 1}, input_pos={1}
   → dynamic_shapes: None (all static)
   → builder._export() → torch.export.export()

4. Lower both together
   → to_edge_transform_and_lower({"kv_forward": ..., "decode": ...})
   → XNNPACK partitioner delegates eligible ops
   → ConvertToLinearPass optimizes matmuls

5. Memory planning
   → MemoryPlanningPass(share_mutable_buffers=True)
   → KV cache buffers get identical offsets across both methods (mem_id=2)

6. Serialize
   → save_pte_program() → flatbuffer .pte file
```

## Step 2: Run

```bash
./exe_llama_runner.sh \
  --model llama3_2_prefill_decode.pte \
  --tokenizer ~/.llama/checkpoints/Llama3.2-1B/tokenizer.model \
  --prompt "What is the capital of France?" \
  --seq_len 128 \
  --method_name decode \
  --prefill_method_name kv_forward
```

Or directly with `llama_main`:

```bash
./cmake-out/examples/models/llama/llama_main \
  --model_path llama3_2_prefill_decode.pte \
  --tokenizer_path ~/.llama/checkpoints/Llama3.2-1B/tokenizer.model \
  --prompt "What is the capital of France?" \
  --seq_len 128 \
  --temperature 0 \
  --method_name decode \
  --prefill_method_name kv_forward
```

### Runtime behavior

1. **Load**: Both methods loaded with a shared `HierarchicalAllocator` — KV
   cache lives in `buffer[2]` at identical offsets for both methods.

2. **Prefill** (`kv_forward`): All prompt tokens processed in parallel.
   ```
   tokens: {1, N}    input_pos: [0, 1, ..., N-1]
   → Writes KV cache positions 0..N-1
   → Returns logits for next-token prediction
   ```

3. **Decode** (`decode`): Tokens generated one at a time.
   ```
   tokens: {1, 1}    input_pos: [N]
   → Reads KV cache 0..N-1 (written by kv_forward)
   → Writes position N
   → Returns logits → sample → repeat with N+1, N+2, ...
   ```

## Key design decisions

### Why `enable_dynamic_shape=False` in model config?

When `True`, the model's rope and attention code use `.item()` to extract scalar
positions. This fails during `torch.export` tracing:
```
RuntimeError: Cannot cast FakeTensor to number
```

Setting it `False` uses direct tensor indexing (`freqs_cos[input_pos]`) which
works with both static and dynamic `torch.export` shapes. The `dynamic_shapes`
parameter to `torch.export.export()` is a separate mechanism.

### Why `metadata["enable_dynamic_shape"] = True`?

This runtime metadata tells the C++ runner to use **parallel prefill** — sending
all prompt tokens at once as `{1, N}` instead of one-by-one as `{1, 1}`.

### Why `share_mutable_buffers=True`?

Without it, each method's KV cache gets planned at different memory offsets.
Even though both methods share the same allocator at runtime, kv_forward would
write to offset X while decode reads from offset Y — producing garbage.

With it, the memory planner assigns **identical offsets** to buffers with the
same fully-qualified name across methods. Both methods read/write the exact same
physical memory addresses.

## Customization

### Different max sequence length

```bash
python export_llama_prefill_decode.py \
  -c ... -p ... --max_seq_length 512 --use_sdpa_with_kv_cache -X -kv
```

Larger `max_seq_length` increases KV cache size and .pte file size.

### Without XNNPACK delegation

Remove `-X` to run entirely on the portable (CPU) backend:

```bash
python export_llama_prefill_decode.py \
  -c ... -p ... --max_seq_length 128 --use_sdpa_with_kv_cache -kv
```

Smaller .pte file, but slower inference.

### Different checkpoint

Works with any Llama-architecture model. Adjust `-c` and `-p`:

```bash
python export_llama_prefill_decode.py \
  -c /path/to/model.pth \
  -p /path/to/params.json \
  --max_seq_length 128 --use_sdpa_with_kv_cache -X -kv
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Cannot cast FakeTensor to number` | `enable_dynamic_shape=True` in model | Script already sets it `False` — don't override |
| Garbage output from decode | KV cache not shared | Verify `share_mutable_buffers=True` |
| `ConstraintViolationError` | Dynamic dim range issue | Ensure `Dim("seq", min=1, max=...)` with `min=1` |
| OOM during export | Model too large | Reduce `--max_seq_length` or use quantization flags |
| `llama_main` not found | Runner not built | Run `./build_llama_runner.sh` first |
