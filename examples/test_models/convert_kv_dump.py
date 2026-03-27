#!/usr/bin/env python3
"""
Convert KV cache binary dump to readable numpy format.

Usage:
    python3 convert_kv_dump.py <filename.bin> <cache_len> [--head_dim 128] [--n_kv_heads 8]

Examples:
    python3 convert_kv_dump.py prefill_k_input_layer0_pos0_step0.bin 896
    python3 convert_kv_dump.py decode_k_input_layer0_pos128_step1.bin 1023 --head_dim 128 --n_kv_heads 8
"""

import argparse
import os
import numpy as np


def convert_kv_dump(filename, cache_len, n_kv_heads=None, head_dim=None):
    data = np.fromfile(filename, dtype=np.float32)
    total = data.size

    if n_kv_heads and head_dim:
        expected = n_kv_heads * cache_len * head_dim
        if total != expected:
            print(f"Warning: total elements ({total}) != "
                  f"n_kv_heads({n_kv_heads}) x cache_len({cache_len}) x head_dim({head_dim}) = {expected}")
    elif head_dim:
        n_kv_heads = total // (cache_len * head_dim)
        print(f"Auto-detected n_kv_heads = {n_kv_heads}")
    elif n_kv_heads:
        head_dim = total // (cache_len * n_kv_heads)
        print(f"Auto-detected head_dim = {head_dim}")
    else:
        head_dim = 128
        n_kv_heads = total // (cache_len * head_dim)
        print(f"Auto-detected n_kv_heads = {n_kv_heads}, head_dim = {head_dim}")

    data = data.reshape(n_kv_heads, cache_len, head_dim)

    base = os.path.splitext(filename)[0]

    npy_path = base + ".npy"
    np.save(npy_path, data)
    print(f"Saved: {npy_path}  shape={data.shape}  dtype={data.dtype}")

    txt_path = base + ".txt"
    with open(txt_path, "w") as f:
        f.write(f"shape: {data.shape}\n")
        f.write(f"dtype: {data.dtype}\n")
        f.write(f"min: {data.min():.6f}  max: {data.max():.6f}  mean: {data.mean():.6f}\n\n")
        for h in range(n_kv_heads):
            nonzero = np.count_nonzero(data[h])
            f.write(f"--- head {h} (nonzero: {nonzero}/{cache_len * head_dim}) ---\n")
            for s in range(min(cache_len, 8)):
                vals = " ".join(f"{v:10.6f}" for v in data[h, s, :8])
                f.write(f"  seq[{s:4d}]: {vals} ...\n")
            if cache_len > 8:
                f.write(f"  ... ({cache_len - 8} more rows)\n")
            f.write("\n")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert KV cache dump to npy/txt")
    parser.add_argument("filename", help="Input .bin file")
    parser.add_argument("cache_len", type=int, help="Cache sequence length")
    parser.add_argument("--head_dim", type=int, default=None)
    parser.add_argument("--n_kv_heads", type=int, default=None)
    args = parser.parse_args()

    convert_kv_dump(args.filename, args.cache_len, args.n_kv_heads, args.head_dim)
