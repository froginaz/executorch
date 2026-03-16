# KV-Cache Tensor Shape and Management Guide

## 1. What is KV-Cache?

Transformer attention computes Key and Value projections for every token. During autoregressive generation, previously computed K/V values are cached to avoid redundant computation. The KV-cache stores these values and grows with each new token.

## 2. Tensor Shapes

ExecuTorch supports two cache layouts controlled by `transpose_cache`:

### Transposed Layout (`transpose_cache=True`)

```
k_cache: [batch_size, num_kv_heads, max_seq_len, head_dim]
v_cache: [batch_size, num_kv_heads, max_seq_len, head_dim]

             B=1    H=32   S=2048  D=128
k_cache = zeros(1,    32,   2048,   128)   # ~32MB per cache (fp32)
v_cache = zeros(1,    32,   2048,   128)
```

### Non-Transposed Layout (`transpose_cache=False`)

```
k_cache: [batch_size, max_seq_len, num_kv_heads, head_dim]
v_cache: [batch_size, max_seq_len, num_kv_heads, head_dim]

             B=1   S=2048  H=32   D=128
k_cache = zeros(1,  2048,   32,    128)
v_cache = zeros(1,  2048,   32,    128)
```

### Dimension Summary

| Dim | Name | Typical Values | Description |
|-----|------|----------------|-------------|
| B | batch_size | 1 | Number of sequences (usually 1 for on-device) |
| H | num_kv_heads | 8, 32 | Number of KV attention heads (may differ from Q heads in GQA) |
| S | max_seq_len | 128, 2048, 8192 | Maximum sequence length the cache can hold |
| D | head_dim | 64, 128 | Dimension per attention head |

### Per-Layer Cache

Each transformer layer has its own independent KV-cache pair. For a 32-layer model:

```
layer_0: k_cache[1, 32, 2048, 128], v_cache[1, 32, 2048, 128]
layer_1: k_cache[1, 32, 2048, 128], v_cache[1, 32, 2048, 128]
...
layer_31: k_cache[1, 32, 2048, 128], v_cache[1, 32, 2048, 128]

Total KV-cache memory = 32 layers x 2 (K+V) x 1x32x2048x128 x 4 bytes
                      = ~2 GB (fp32)
                      = ~1 GB (fp16)
```

## 3. Position Tracking

The cache tracks which positions have been written using `kv_cache_pos`:

```python
# Initialized as sequential indices
kv_cache_pos = [0, 1, 2, 3, 4, ..., max_seq_len-1]
```

After writing `seq_len` tokens, the position tracker advances:

```python
self.kv_cache_pos.add_(seq_len)
```

This avoids dynamic control flow, keeping the logic export-friendly for ExecuTorch.

## 4. Cache Update During Prefill

Prefill processes multiple tokens in parallel (e.g., the entire prompt).

```
Input prompt: "The quick brown fox" → tokens = [The, quick, brown, fox]
seq_len = 4
```

### Before Prefill

```
kv_cache_pos = [0, 1, 2, 3, 4, 5, ..., 2047]

k_cache (seq dim):
  pos:  [  0  |  1  |  2  |  3  |  4  |  5  | ... | 2047 ]
  data: [  0  |  0  |  0  |  0  |  0  |  0  | ... |  0   ]
```

### Prefill Step (seq_len=4)

```python
# New K/V from attention projection
k_val.shape = [1, 32, 4, 128]   # 4 tokens at once
v_val.shape = [1, 32, 4, 128]

# Write to cache at positions kv_cache_pos[:4] → [0, 1, 2, 3]
k_cache[:, :, kv_cache_pos[:4]] = k_val
v_cache[:, :, kv_cache_pos[:4]] = v_val

# Advance position tracker
kv_cache_pos.add_(4)
```

### After Prefill

```
kv_cache_pos = [4, 5, 6, 7, 8, 9, ..., 2051]

k_cache (seq dim):
  pos:  [  0   |  1     |  2    |  3  |  4  |  5  | ... | 2047 ]
  data: [ The  | quick  | brown | fox |  0  |  0  | ... |  0   ]
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         filled by prefill
```

## 5. Cache Update During Decode

Decode processes one token at a time (autoregressive generation).

### Decode Step 1: Generate token 5

```python
# New K/V from attention projection for 1 token
k_val.shape = [1, 32, 1, 128]
v_val.shape = [1, 32, 1, 128]

# Write to cache at position kv_cache_pos[:1] → [4]
k_cache[:, :, kv_cache_pos[:1]] = k_val
v_cache[:, :, kv_cache_pos[:1]] = v_val

# Advance position tracker
kv_cache_pos.add_(1)
```

### After Decode Step 1

```
kv_cache_pos = [5, 6, 7, 8, 9, 10, ..., 2052]

k_cache (seq dim):
  pos:  [  0   |  1     |  2    |  3  |   4    |  5  | ... | 2047 ]
  data: [ The  | quick  | brown | fox | jumps  |  0  | ... |  0   ]
                                        ^^^^^
                                        filled by decode step 1
```

### Decode Step 2: Generate token 6

```
kv_cache_pos[:1] → [5]

After:
  pos:  [  0   |  1     |  2    |  3  |   4    |   5   | 6 | ... ]
  data: [ The  | quick  | brown | fox | jumps  | over  | 0 | ... ]
                                                 ^^^^
                                                 decode step 2
```

## 6. Full Sequence Diagram

```
Step 0 - Initial State:
  cache: [ _  _  _  _  _  _  _  _  ]    pos_tracker: [0,1,2,3,4,5,6,7]

Step 1 - Prefill "Hello world !" (3 tokens):
  cache: [ H  W  !  _  _  _  _  _  ]    pos_tracker: [3,4,5,6,7,8,9,10]
                                          ↑ advanced by 3

Step 2 - Decode → "How" (1 token):
  cache: [ H  W  !  How  _  _  _  _ ]   pos_tracker: [4,5,6,7,8,9,10,11]
                                          ↑ advanced by 1

Step 3 - Decode → "are" (1 token):
  cache: [ H  W  !  How are  _  _  _ ]  pos_tracker: [5,6,7,8,9,10,11,12]

Step 4 - Decode → "you" (1 token):
  cache: [ H  W  !  How are you  _  _ ] pos_tracker: [6,7,8,9,10,11,12,13]
```

## 7. Alternative Update Method: index_copy_

When `enable_dynamic_shape=True`, the cache uses `index_copy_` instead of fancy indexing:

```python
def update(self, input_pos, k_val, v_val):
    start_pos = input_pos[0].item()
    seq_length = k_val.size(2)
    indices = torch.arange(seq_length) + start_pos

    # Prefill: indices = [0, 1, 2, 3]  (start_pos=0, seq_length=4)
    # Decode:  indices = [4]            (start_pos=4, seq_length=1)

    self.k_cache.index_copy_(2, indices, k_val)  # dim=2 is seq_len
    self.v_cache.index_copy_(2, indices, v_val)
```

## 8. Runner-Managed Cache (External)

When the C++ runner manages the cache externally, the model uses `scatter` to combine old cache with new values:

```python
# Model receives k_cache as input (from runner)
layer_k = k_cache[self.layer_id]          # [B, H, S_max, D]
indices = input_pos.reshape(1, 1, -1, 1).expand_as(k)  # broadcast

# Scatter new K values into the cache
k = layer_k.scatter(2, indices, k)        # old cache + new values at input_pos
v = layer_v.scatter(2, indices, v)

# Runner receives updated k, v as outputs and feeds back next step
```

In this mode:
- The runner (C++) holds `k_cache` and `v_cache` tensors
- Each step: runner passes cache as input → model returns updated cache → runner stores it
- No mutable buffer sharing needed in the PTE

## 9. Shared KV-Cache Between Prefill and Decode (ExecuTorch)

When prefill and decode are separate methods in one PTE, they share KV-cache via shared mutable buffers:

```
┌─────────────────────────────────────────────────────────┐
│                    PTE File                              │
│                                                         │
│  ┌─────────────┐              ┌──────────────┐         │
│  │   prefill    │              │    decode     │         │
│  │  method      │              │   method      │         │
│  └──────┬───────┘              └──────┬────────┘         │
│         │ write                       │ read + write     │
│         ▼                             ▼                  │
│  ┌──────────────────────────────────────────────┐       │
│  │         Shared Planned Memory (mem_id=2)      │       │
│  │                                               │       │
│  │   k_cache [B, H, S_max, D]  (all layers)     │       │
│  │   v_cache [B, H, S_max, D]  (all layers)     │       │
│  │   kv_cache_pos [S_max]                        │       │
│  └──────────────────────────────────────────────┘       │
│         ↑ same physical memory (HierarchicalAllocator)   │
└─────────────────────────────────────────────────────────┘
```

### C++ Runtime Setup

```cpp
// Both methods share the same HierarchicalAllocator
auto shared = create_shared_planned_memory(prefill_meta, decode_meta);

MemoryManager prefill_mm(&prefill_allocator, shared.allocator.get());
MemoryManager decode_mm(&decode_allocator, shared.allocator.get());

program->load_method("prefill", &prefill_mm);
program->load_method("decode",  &decode_mm);

// Execute
prefill_method->execute();   // Writes KV-cache at positions [0..prompt_len-1]
decode_method->execute();    // Reads existing cache, writes at position [prompt_len]
decode_method->execute();    // Reads cache, writes at [prompt_len+1]
// ... continues autoregressively
```

## 10. Exporting a Prefill/Decode Model with Shared KV-Cache

### Step 1: Export Two Methods with Different Shapes

```python
import torch
from torch.export import Dim, export

model = MyLlamaModel(use_kv_cache=True).eval()
max_seq_len = 2048

# --- Prefill: dynamic sequence length [1..max_seq_len] ---
seq_dim = Dim("seq", min=1, max=max_seq_len)

prefill_inputs = (
    torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0),  # [1, S]
    torch.arange(0, max_seq_len, dtype=torch.long),                    # input_pos [S]
)
prefill_dynamic_shapes = (
    {1: seq_dim},           # tokens dim 1 is dynamic
    {0: seq_dim},           # input_pos dim 0 is dynamic
)
prefill_prog = export(model, prefill_inputs, dynamic_shapes=prefill_dynamic_shapes)

# --- Decode: fixed single token ---
decode_inputs = (
    torch.tensor([[1]], dtype=torch.long),     # [1, 1]
    torch.tensor([0], dtype=torch.long),       # input_pos [1]
)
decode_prog = export(model, decode_inputs)     # No dynamic shapes
```

### Step 2: Lower Both Methods Together

```python
from executorch.exir import to_edge_transform_and_lower

edge_manager = to_edge_transform_and_lower(
    {"kv_forward": prefill_prog, "decode": decode_prog},
    compile_config=edge_config,
    partitioner=partitioners,       # e.g., XnnpackPartitioner
    constant_methods={
        "get_max_seq_len": max_seq_len,
        "use_kv_cache": True,
        "enable_dynamic_shape": True,
    },
)
```

`constant_methods` adds lightweight methods to the PTE that return config values at runtime.

### Step 3: Serialize with Shared Mutable Buffers

```python
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass

export_program = edge_manager.to_executorch(
    ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=False,
            share_mutable_buffers=True,    # KV-cache shared between methods
        ),
        sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
    ),
    additional_passes=[
        InitializedMutableBufferPass(["kv_cache_pos"]),  # Initialize position tracker
    ],
)

with open("llama_prefill_decode.pte", "wb") as f:
    f.write(export_program.buffer)
```

### What Each Option Does

| Option | Effect |
|--------|--------|
| `share_mutable_buffers=True` | KV-cache tensors get `mem_id=2` with fixed offsets. Both methods write to the same physical memory at runtime. |
| `alloc_graph_input=False` | Graph inputs (tokens, input_pos) are provided by the runner, not pre-allocated in the PTE. |
| `InitializedMutableBufferPass(["kv_cache_pos"])` | Saves the initial `[0, 1, 2, ..., max_seq_len-1]` values of `kv_cache_pos` into the PTE so the position tracker starts correctly. |
| `ConstraintBasedSymShapeEvalPass` | Resolves symbolic shapes (from `Dim("seq", ...)`) to concrete upper bounds for memory planning. |

### What Changes in the PTE File

```
Without share_mutable_buffers:
  non_const_buffer_sizes = [0, activation_size]        # 2 entries
  KV-cache tensors: mem_id=1, offsets from greedy algo (may overlap)

With share_mutable_buffers:
  non_const_buffer_sizes = [0, activation_size, kv_cache_size]  # 3 entries
  KV-cache tensors: mem_id=2, sequential non-overlapping offsets
  Both kv_forward and decode have identical mem_id=2 offsets
```

### Using export_llama.py (Convenience Script)

```bash
python -m examples.models.llama.export_llama \
    --model llama3 \
    --checkpoint path/to/checkpoint.pth \
    --params path/to/params.json \
    -kv \
    --prefill_dynamic \
    --xnnpack \
    --output_name llama3_prefill_decode.pte
```

Key flags:
- `-kv`: Enable KV-cache
- `--prefill_dynamic`: Export separate prefill (dynamic seq) and decode (seq=1) methods

## 11. Memory Size Calculation

```
Per layer KV-cache size:
  = 2 (K+V) x batch_size x num_kv_heads x max_seq_len x head_dim x dtype_bytes

Example: Llama 3 8B (GQA with 8 KV heads, 128 head_dim, 32 layers)
  Per layer  = 2 x 1 x 8 x 2048 x 128 x 2 (fp16) = 8 MB
  Total      = 32 layers x 8 MB = 256 MB

Example: Llama 3 70B (GQA with 8 KV heads, 128 head_dim, 80 layers)
  Per layer  = 2 x 1 x 8 x 2048 x 128 x 2 (fp16) = 8 MB
  Total      = 80 layers x 8 MB = 640 MB
```
