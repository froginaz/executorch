# Mamba SSM Runner for ExecuTorch

A C++ runner that loads and executes a Mamba State Space Model exported as `.pte`.

## Overview

Mamba is an attention-free architecture based on Selective State Space Models (SSM).
Unlike Transformer-based runners, it does not require KV cache or attention masks.
Instead, it maintains a fixed-size hidden state that gets updated each step.

```
Transformer: tokens + attn_mask + KV cache (grows) → logits + KV cache
Mamba:       tokens + ssm_state (fixed size)       → logits + ssm_state
```

### Model IO

```
Input:
  [0] tokens     - int64 (1, seq_len)
  [1] ssm_states - float (n_layers, d_inner, d_state)

Output:
  [0] logits     - float (1, seq_len, vocab_size)
  [1] ssm_states - float (n_layers, d_inner, d_state)
```

## Step 1: Export .pte Model

### Prerequisites

```bash
pip install torch executorch
```

### Export (portable)

```bash
cd examples/test_models
python3 export_mamba.py
```

Output: `mamba.pte`

### Export with XNNPACK delegation

```bash
python3 export_mamba.py --xnnpack
```

Output: `mamba_xnnpack.pte`

### Default model config

| Parameter | Value |
|-----------|-------|
| vocab_size | 256 |
| d_model | 64 |
| d_inner | 128 |
| d_state | 16 |
| n_layers | 4 |

## Step 2: Build ExecuTorch Runtime

Build the ExecuTorch libraries first:

```bash
cd /path/to/executorch

# Install dependencies
./install_executorch.sh

# Build core runtime + portable ops
cmake --preset linux -B build_et
cmake --build build_et -j$(nproc) --target \
    executorch \
    portable_ops_lib \
    portable_kernels \
    extension_data_loader \
    extension_runner_util \
    gflags
```

For XNNPACK support, add:

```bash
cmake --build build_et -j$(nproc) --target xnnpack_backend
```

## Step 3: Build Mamba Runner

```bash
cmake -S examples/test_models -B build_mamba \
    -DCMAKE_PREFIX_PATH="$(pwd)/build_et" \
    -DEXECUTORCH_ROOT="$(pwd)"

cmake --build build_mamba -j$(nproc)
```

## Step 4: Run

### Basic usage

```bash
./build_mamba/mamba_runner \
    --model_path=examples/test_models/mamba.pte \
    --prompt="1,2,3" \
    --max_tokens=20
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | `mamba.pte` | Path to the .pte model file |
| `--prompt` | `1,2,3` | Comma-separated token IDs |
| `--max_tokens` | `20` | Number of tokens to generate |
| `--n_layers` | `4` | Number of SSM layers |
| `--d_inner` | `128` | SSM inner dimension |
| `--d_state` | `16` | SSM state dimension |

### Example output

```
I 00:00:00.000 Method: forward
I 00:00:00.001 Prompt tokens: 3
I 00:00:00.001 --- Prefill ---
I 00:00:00.005 Prefill [0]: tok=1 -> next=42
I 00:00:00.008 Prefill [1]: tok=2 -> next=87
I 00:00:00.011 Prefill [2]: tok=3 -> next=155
I 00:00:00.011 --- Decode ---
I 00:00:00.014 Decode [0]: tok=155 -> next=203
I 00:00:00.017 Decode [1]: tok=203 -> next=91
...

Generated tokens: [155, 203, 91, ...]
```

## Files

| File | Description |
|------|-------------|
| `mamba_runner.cpp` | C++ runner executable |
| `CMakeLists.txt` | Build configuration |
| `export_mamba.py` | Python export script |
| `mamba.pte` | Pre-exported model (portable) |

## Using a Real Mamba Model

To use a HuggingFace Mamba model (e.g., `state-spaces/mamba-130m-hf`),
modify `export_mamba.py` to load the pretrained weights:

```python
from transformers import MambaForCausalLM

model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
```

Then adjust the runner flags to match the model dimensions:

```bash
./mamba_runner \
    --model_path=mamba_130m.pte \
    --n_layers=24 --d_inner=1536 --d_state=16 \
    --max_tokens=50
```
