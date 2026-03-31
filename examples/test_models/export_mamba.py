"""
Export a Mamba SSM model to .pte with XNNPACK delegation.

Mamba uses selective state space model (SSM) instead of attention.
No KV cache or attention mask needed - only a fixed-size hidden state.

Input:  [tokens(1, seq_len), ssm_state(n_layers, d_inner, d_state)]
Output: [logits(1, seq_len, vocab_size), ssm_state(n_layers, d_inner, d_state)]

Usage:
    python3 export_mamba.py
    python3 export_mamba.py --xnnpack
"""

import argparse
import torch
import torch.nn as nn
from torch.export import export


class MambaSSMBlock(nn.Module):
    def __init__(self, d_model, d_inner, d_state):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d_weight = nn.Parameter(torch.randn(d_inner, 1, 4))
        self.conv1d_bias = nn.Parameter(torch.zeros(d_inner))

        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, d_inner, bias=True)

        self.A = nn.Parameter(torch.randn(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x, ssm_state):
        # x: (B, L, d_model), ssm_state: (B, d_inner, d_state)
        bsz, seq_len, _ = x.shape

        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_part, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Simplified conv1d (causal, kernel=4) via F.conv1d
        x_conv = x_part.transpose(1, 2)  # (B, d_inner, L)
        x_conv = torch.nn.functional.pad(x_conv, (3, 0))
        x_conv = torch.nn.functional.conv1d(
            x_conv, self.conv1d_weight, self.conv1d_bias, groups=self.d_inner)
        x_conv = torch.nn.functional.silu(x_conv).transpose(1, 2)  # (B, L, d_inner)

        # SSM parameters from input
        x_db = self.x_proj(x_conv)  # (B, L, 2*d_state)
        B_param, C_param = x_db.chunk(2, dim=-1)  # each (B, L, d_state)
        dt = torch.nn.functional.softplus(self.dt_proj(B_param))  # (B, L, d_inner)

        # Discretize A
        A = -torch.exp(self.A)  # (d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_param.unsqueeze(2)  # (B, L, d_inner, d_state)

        # Sequential scan
        ys = []
        h = ssm_state
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x_conv[:, t].unsqueeze(-1)
            y_t = (h * C_param[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        output = self.out_proj(y * torch.nn.functional.silu(z))
        return output, h


class MambaModel(nn.Module):
    def __init__(self, vocab_size, d_model, d_inner, d_state, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.d_inner = d_inner
        self.d_state = d_state

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [MambaSSMBlock(d_model, d_inner, d_state) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens, ssm_states):
        # tokens: (1, seq_len), ssm_states: (n_layers, d_inner, d_state)
        x = self.embed(tokens)

        new_states = []
        for i, layer in enumerate(self.layers):
            state_i = ssm_states[i].unsqueeze(0)  # (1, d_inner, d_state)
            x, new_state = layer(x, state_i)
            new_states.append(new_state.squeeze(0))

        x = self.norm(x)
        logits = self.lm_head(x)
        out_states = torch.stack(new_states, dim=0)  # (n_layers, d_inner, d_state)
        return logits, out_states


def export_mamba(use_xnnpack=False):
    vocab_size = 256
    d_model = 64
    d_inner = 128
    d_state = 16
    n_layers = 4
    seq_len = 1

    model = MambaModel(vocab_size, d_model, d_inner, d_state, n_layers)
    model.eval()

    tokens = torch.randint(0, vocab_size, (1, seq_len))
    ssm_states = torch.zeros(n_layers, d_inner, d_state)

    print(f"Exporting Mamba model (xnnpack={use_xnnpack})...")
    print(f"  vocab={vocab_size}, d_model={d_model}, d_inner={d_inner}, "
          f"d_state={d_state}, n_layers={n_layers}")
    print(f"  Input:  tokens(1,{seq_len}), ssm_states({n_layers},{d_inner},{d_state})")
    print(f"  Output: logits(1,{seq_len},{vocab_size}), ssm_states({n_layers},{d_inner},{d_state})")

    with torch.no_grad():
        exported = export(model, (tokens, ssm_states))

    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

    partitioners = []
    if use_xnnpack:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        partitioners.append(XnnpackPartitioner())

    edge = to_edge_transform_and_lower(
        exported,
        partitioner=partitioners,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    et_program = edge.to_executorch()

    suffix = "_xnnpack" if use_xnnpack else ""
    output_path = f"mamba{suffix}.pte"
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    print(f"Exported to {output_path} ({len(et_program.buffer)} bytes)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xnnpack", action="store_true")
    args = parser.parse_args()
    export_mamba(use_xnnpack=args.xnnpack)
