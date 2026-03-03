# How Input Tensors Are Constructed from Tokens

Traces the path from raw text to the embedding layer input in the ExecuTorch
Llama runner.

## Pipeline overview

```
"The sky is"
     │
     ▼  tokenizer.encode(bos=1)
[128000, 791, 13180, 374]              std::vector<uint64_t>
     │
     ▼  from_blob({1, N}, Long)
[[128000, 791, 13180, 374]]            Tensor {1, 4} int64
     │
     ▼  nn.Embedding(128256, 2048)
[[[0.12, -0.34, …],                    Tensor {1, 4, 2048} float32
  [0.56,  0.78, …],                      → rope → attention → FFN → logits
  [-0.11, 0.45, …],
  [0.33, -0.67, …]]]
```

## Step 1: Text → token IDs (tokenizer)

The tokenizer converts a string into a sequence of integer token IDs.

C++ (`text_llm_runner.cpp:122-135`):
```cpp
auto encode_res = tokenizer_->encode(prompt, /*bos=*/config.num_bos, /*eos=*/config.num_eos);
prompt_tokens = encode_res.get();
// "The sky is" with bos=1 → [128000, 791, 13180, 374]
//                              ^^^^^^ BOS token prepended
```

- `bos` (beginning of sequence) tokens are prepended by the tokenizer
- `eos` (end of sequence) tokens are appended if requested
- Token IDs are `uint64_t` integers indexing into the vocabulary

## Step 2: Token IDs → input tensor

The vector of token IDs is wrapped into a 2D tensor with `from_blob()`.
The shape depends on whether we are in prefill or decode phase.

### Prefill (parallel) — `text_prefiller.cpp:83-86`

All prompt tokens packed into one tensor:

```cpp
auto tokens = from_blob(
    prompt_tokens.data(),
    {1, num_prompt_tokens},        // shape: {batch=1, seq_len=N}
    executorch::aten::ScalarType::Long
);
// Example: [[128000, 791, 13180, 374]]  shape {1, 4}
```

`from_blob` creates a tensor that **points directly to the vector's memory** —
no copy. The tensor is a view over `prompt_tokens.data()`.

### Decode (sequential) — `text_token_generator.h:72-84`

One token at a time:

```cpp
token_data = {cur_token};          // single token ID
token_shape = {1, 1};              // shape: {batch=1, seq_len=1}
auto tokens_managed = from_blob(
    token_data.data(), token_shape, executorch::aten::ScalarType::Long
);
// Example: [[42]]  shape {1, 1}
```

On each decode step, `token_data[0]` is updated with the newly predicted token:
```cpp
token_data[0] = cur_token;  // reuse the same tensor, just update the value
```

### Shape summary

| Phase | Tensor shape | Contents |
|-------|-------------|----------|
| Prefill (kv_forward) | `{1, N}` where N = num_prompt_tokens | All prompt token IDs |
| Decode | `{1, 1}` | Single predicted token ID |

The first dimension is always 1 (batch size). The second dimension is the
sequence length — variable for prefill (when using dynamic shapes), fixed at 1
for decode.

## Step 3: Token tensor → embedding lookup (model)

The token tensor enters the model's `forward()` method.

`llama_transformer.py:180-191`:
```python
def forward(self, tokens, attn_options=None, h=None):
    if self.apply_embedding and tokens is not None:
        h = self.tok_embeddings(tokens)   # nn.Embedding lookup
```

`tok_embeddings` is `nn.Embedding(vocab_size, dim)` — a weight matrix of shape
`{vocab_size, dim}` (e.g., `{128256, 2048}` for Llama 3.2 1B).

The embedding operation is a **pure table lookup**: each token ID selects one
row from the weight matrix.

```
tokens: [[128000, 791, 13180, 374]]     {1, 4} int64
              │      │     │      │
              ▼      ▼     ▼      ▼
         weight[128000]  weight[791]  weight[13180]  weight[374]
              │      │     │      │
              ▼      ▼     ▼      ▼
h:      [[[0.12, ...],                   {1, 4, 2048} float32
          [0.56, ...],
          [-0.11, ...],
          [0.33, ...]]]
```

No matrix multiplication — just `weight[token_id]` for each token. The output
shape is `{batch, seq_len, embedding_dim}`.

## Step 4: Alongside the token tensor — input_pos

The model also receives position information as the second input. This tells
the KV cache and rope where these tokens sit in the sequence.

`text_decoder_runner.cpp:50-56` → `util.h:111-150`:
```cpp
auto start_pos_tensor = populate_start_pos_or_cache_position(
    module_, start_pos, cache_positions, tokens->numel(), method_name_.c_str());
```

The function reads the method's metadata to decide the format:

**kv_forward** (dynamic, `input_tensor_meta(1).sizes()[0] > 1`):
```cpp
// Range tensor: [start_pos, start_pos+1, ..., start_pos+N-1]
cache_positions_vec = {0, 1, 2, 3};   // for start_pos=0, seq_len=4
// shape: {4}
```

**decode** (fixed, `input_tensor_meta(1).sizes()[0] == 1`):
```cpp
// Scalar tensor: [start_pos]
// shape: {1}
```

## Complete input construction

Both inputs are assembled and passed to `module_->execute()`:

```cpp
// text_decoder_runner.cpp:62-67
auto inputs_res = io_manager_->prepare_decode(tokens, start_pos_tensor, method_name_);
inputs = inputs_res.get();
auto outputs_res = module_->execute(method_name_, inputs);
```

The model receives:
```
Input 0 (tokens):    [[128000, 791, 13180, 374]]   {1, 4}  int64
Input 1 (input_pos): [0, 1, 2, 3]                  {4}     int64
```

Which maps to the Python forward signature:
```python
def forward(self, tokens, attn_options):
    # tokens = tensor([[128000, 791, 13180, 374]])
    # attn_options = {"input_pos": tensor([0, 1, 2, 3])}
```

## Export-time example inputs

When exporting, the example inputs must match this structure.

**kv_forward** (dynamic shapes):
```python
builder.example_inputs = (
    torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0),  # {1, 128}
    {"input_pos": torch.arange(0, max_seq_len, dtype=torch.long)},     # {128}
)
builder.dynamic_shapes = (
    {1: Dim("seq", min=1, max=max_seq_len)},
    {"input_pos": {0: Dim("seq", min=1, max=max_seq_len)}},
)
```

**decode** (fixed shapes):
```python
builder.example_inputs = (
    torch.tensor([[1]], dtype=torch.long),              # {1, 1}
    {"input_pos": torch.tensor([0], dtype=torch.long)}, # {1}
)
builder.dynamic_shapes = None
```

The dict wrapper `{"input_pos": ...}` corresponds to the `attn_options`
parameter in the model's forward signature. During `torch.export`, this dict
is flattened — the exported method's second input is the `input_pos` tensor
directly.


---
Let me trace the full path from raw text to the embedding layer input.                                                                                                                               
                                                                                                                                                                                                       
  Token → Embedding Input Tensor                                                                                                                                                                       
                                                                                                                                                                                                       
  Step 1: Text → Token IDs (Tokenizer)                                                                                                                                                                 

  "The sky is" → tokenizer.encode() → [791, 13180, 374]                                                                                                                                                
   
  In C++ (text_llm_runner.cpp:122-135):                                                                                                                                                                
  auto encode_res = tokenizer_->encode(prompt, /*bos=*/1, /*eos=*/0);                                                                                                                                
  prompt_tokens = encode_res.get();  // std::vector<uint64_t> = [128000, 791, 13180, 374]
  //                                                              ^^^^^^ BOS token

  Step 2: Token IDs → Input Tensor (TextPrefiller / TextTokenGenerator)

  Prefill (parallel, all tokens at once) — text_prefiller.cpp:83-86:
  auto tokens = from_blob(
      prompt_tokens.data(),
      {1, num_prompt_tokens},        // shape: {1, 4}  (batch=1, seq_len=4)
      executorch::aten::ScalarType::Long
  );
  // Tensor content: [[128000, 791, 13180, 374]]
  //                    BOS    The   sky    is

  Decode (one token at a time) — text_token_generator.h:75-76:
  token_data = {cur_token};          // e.g. {next_predicted_token}
  token_shape = {1, 1};              // shape: {1, 1}  (batch=1, seq_len=1)
  auto tokens_managed = from_blob(
      token_data.data(), token_shape, executorch::aten::ScalarType::Long
  );
  // Tensor content: [[42]]   (one token ID)

  Step 3: Token Tensor → Embedding (Model forward)

  The token tensor enters the model's forward() in llama_transformer.py:180-191:

  def forward(self, tokens, attn_options=None, h=None):
      if self.apply_embedding and tokens is not None:
          h = self.tok_embeddings(tokens)   # ← embedding lookup

  tok_embeddings is nn.Embedding(vocab_size, dim):

  tokens: [[128000, 791, 13180, 374]]     shape: {1, 4}  (int64)
           │        │     │      │
           ▼        ▼     ▼      ▼
      ┌─────────────────────────────────┐
      │  nn.Embedding(128256, 2048)     │   weight: {128256, 2048} (float32)
      │                                 │
      │  row 128000 → [0.12, -0.34, …]  │   lookup by index
      │  row 791    → [0.56,  0.78, …]  │
      │  row 13180  → [-0.11, 0.45, …]  │
      │  row 374    → [0.33, -0.67, …]  │
      └─────────────────────────────────┘
           │        │     │      │
           ▼        ▼     ▼      ▼
  h:   [[[ 0.12, -0.34, …],              shape: {1, 4, 2048}  (float32)
         [ 0.56,  0.78, …],
         [-0.11,  0.45, …],
         [ 0.33, -0.67, …]]]

  nn.Embedding is a pure table lookup — each token ID selects one row from the weight matrix. No multiplication, just indexing: weight[token_id].

  Summary

  "The sky is"
       │
       ▼  tokenizer.encode(bos=1)
  [128000, 791, 13180, 374]         std::vector<uint64_t>
       │
       ▼  from_blob({1, N}, Long)
  [[128000, 791, 13180, 374]]       Tensor {1, 4} int64
       │
       ▼  nn.Embedding(128256, 2048)
  [[[0.12, -0.34, …],               Tensor {1, 4, 2048} float32
    [0.56,  0.78, …],                 │
    [-0.11, 0.45, …],                 │  → rope → attention → FFN → ...
    [0.33, -0.67, …]]]                ▼

  The input tensor shape is always {batch_size, seq_len} where:
  - Prefill (kv_forward): seq_len = num_prompt_tokens (variable, 1..max_seq_len)
  - Decode: seq_len = 1 (fixed, one token per step)
