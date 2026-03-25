"""
Export a dummy transformer model matching the grouped KV cache IO layout
used by RunnerManagedCacheIOManager.

Input:  [token, start_pos, attn_mask, k_past_0..k_past_{L-1}, v_past_0..v_past_{L-1}]
Output: [logits, k_out_0..k_out_{L-1}, v_out_0..v_out_{L-1}]
"""

import torch
import torch.nn as nn
from torch.export import export, Dim


class DummyTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, n_kv_heads, head_dim):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_dim, n_kv_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_kv_heads * head_dim, hidden_dim, bias=False)

    def forward(self, x, k_past, v_past):
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        k_out = k_new
        v_out = v_new

        k_full = torch.cat([k_past, k_new], dim=2)
        v_full = torch.cat([v_past, v_new], dim=2)

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k_full.transpose(2, 3)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_full)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.o_proj(attn_output)
        return output, k_out, v_out


class DummyLlamaModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, n_kv_heads, head_dim):
        super().__init__()
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [DummyTransformerLayer(hidden_dim, n_kv_heads, head_dim) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, tokens, start_pos, attn_mask, *kv_past):
        # kv_past: k_past_0, ..., k_past_{L-1}, v_past_0, ..., v_past_{L-1}
        x = self.embed(tokens)

        k_outs = []
        v_outs = []
        for i, layer in enumerate(self.layers):
            k_past_i = kv_past[i]
            v_past_i = kv_past[self.n_layers + i]
            x, k_out, v_out = layer(x, k_past_i, v_past_i)
            k_outs.append(k_out)
            v_outs.append(v_out)

        logits = self.lm_head(x)
        return (logits, *k_outs, *v_outs)


def export_model(mode="decode"):
    vocab_size = 128
    hidden_dim = 64
    n_layers = 2
    n_kv_heads = 2
    head_dim = 32
    max_seq_len = 32

    if mode == "prefill":
        seq_len = 8
        cache_len = max_seq_len - seq_len
        attn_mask_shape = (seq_len, max_seq_len)
    else:
        seq_len = 1
        cache_len = max_seq_len - 1
        attn_mask_shape = (1, max_seq_len)

    model = DummyLlamaModel(vocab_size, hidden_dim, n_layers, n_kv_heads, head_dim)
    model.eval()

    tokens = torch.randint(0, vocab_size, (1, seq_len))
    start_pos = torch.tensor([0], dtype=torch.long)
    attn_mask = torch.ones(attn_mask_shape, dtype=torch.float32)

    kv_past = []
    for _ in range(n_layers):
        kv_past.append(torch.zeros(1, n_kv_heads, cache_len, head_dim))
    for _ in range(n_layers):
        kv_past.append(torch.zeros(1, n_kv_heads, cache_len, head_dim))

    args = (tokens, start_pos, attn_mask, *kv_past)

    print(f"Exporting {mode} model...")
    print(f"  seq_len={seq_len}, cache_len={cache_len}")
    print(f"  n_layers={n_layers}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
    print(f"  Input order: [tokens, start_pos, attn_mask, k0..k{n_layers-1}, v0..v{n_layers-1}]")
    print(f"  Output order: [logits, k0..k{n_layers-1}, v0..v{n_layers-1}]")

    with torch.no_grad():
        exported = export(model, args)

    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    edge = to_edge_transform_and_lower(
        exported,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    et_program = edge.to_executorch()

    output_path = f"/home/user/executorch/dummy_llama_{mode}.pte"
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    print(f"Exported to {output_path}")
    print(f"File size: {len(et_program.buffer)} bytes")
    return output_path


if __name__ == "__main__":
    export_model("prefill")
    export_model("decode")
