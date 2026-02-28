#!/usr/bin/env python3
"""Analyze an ExecuTorch .pte file and print human-readable details."""

import argparse
from collections import Counter

from executorch.exir._serialize._program import deserialize_pte_binary

DTYPE_NAMES = {
    0: "uint8",
    1: "int8",
    2: "int16",
    3: "int32",
    4: "int64",
    5: "float16",
    6: "float32",
    7: "float64",
    10: "bf16",
    11: "qint8",
    12: "quint8",
    13: "qint32",
    15: "quint4x2",
    16: "quint2x4",
}


def dtype_name(scalar_type):
    return DTYPE_NAMES.get(scalar_type, f"dtype_{scalar_type}")


def analyze(path, *, show_trace=False, trace_limit=80):
    with open(path, "rb") as f:
        pte_data = f.read()

    pte_file = deserialize_pte_binary(pte_data)
    program = pte_file.program

    print(f"=== {path} ===")
    print(f"File size: {len(pte_data):,} bytes ({len(pte_data)/1024/1024:.1f} MB)")
    print(f"Execution plans: {len(program.execution_plan)}")

    # Separate execution plans into real methods and metadata.
    method_plans = []
    metadata_plans = []
    for plan in program.execution_plan:
        has_operators = len(plan.operators) > 0
        has_chains = plan.chains and len(plan.chains[0].instructions) > 0
        if has_operators or has_chains:
            method_plans.append(plan)
        else:
            metadata_plans.append(plan)

    # ---- Metadata methods ----
    if metadata_plans:
        print("\n=== METADATA METHODS ===")
        for plan in metadata_plans:
            if plan.values:
                print(f"  {plan.name}(): {plan.values[0].val}")

    # ---- Per-method analysis ----
    for plan in method_plans:
        print(f"\n=== METHOD: {plan.name} ===")
        print(f"  Operators:    {len(plan.operators)}")
        n_instr = len(plan.chains[0].instructions) if plan.chains else 0
        print(f"  Instructions: {n_instr}")
        print(f"  Values:       {len(plan.values)}")
        print(f"  Inputs:       {plan.inputs}")
        print(f"  Outputs:      {plan.outputs}")
        print(f"  Delegates:    {len(plan.delegates)}")
        if len(plan.non_const_buffer_sizes) > 1:
            scratch = plan.non_const_buffer_sizes[1]
            print(f"  Scratch mem:  {scratch/1024/1024:.1f} MB")

        op_names = {i: op.name for i, op in enumerate(plan.operators)}

        # Operator frequency
        if n_instr:
            print(f"\n  --- Operator Frequency ({n_instr} instructions) ---")
            all_ops = Counter()
            for inst in plan.chains[0].instructions:
                ka = inst.instr_args
                if hasattr(ka, "op_index"):
                    all_ops[op_names.get(ka.op_index, "?")] += 1
            for op, cnt in all_ops.most_common():
                print(f"    {op}: {cnt}")

        # Weight memory breakdown
        total_weight_bytes = 0
        weight_groups = {}
        for val in plan.values:
            v = val.val
            if not hasattr(v, "sizes") or not hasattr(v, "data_buffer_idx"):
                continue
            if v.data_buffer_idx == 0:
                continue
            dt = dtype_name(v.scalar_type)
            shape = list(v.sizes)
            buf_size = len(program.constant_buffer[v.data_buffer_idx].storage)
            total_weight_bytes += buf_size
            key = f"{dt} {shape}"
            if key not in weight_groups:
                weight_groups[key] = {"count": 0, "total_bytes": 0}
            weight_groups[key]["count"] += 1
            weight_groups[key]["total_bytes"] += buf_size

        if total_weight_bytes:
            print(f"\n  --- Weight Memory ({total_weight_bytes/1024/1024:.1f} MB) ---")
            for key, info in sorted(
                weight_groups.items(), key=lambda x: -x[1]["total_bytes"]
            ):
                mb = info["total_bytes"] / (1024 * 1024)
                print(f"    {key}: {info['count']}x, {mb:.1f} MB")

        # Dynamic tensors
        alloc_dtypes = Counter()
        for val in plan.values:
            v = val.val
            if hasattr(v, "sizes") and hasattr(v, "allocation_info") and v.allocation_info is not None:
                alloc_dtypes[dtype_name(v.scalar_type)] += 1
        total_alloc = sum(alloc_dtypes.values())
        if total_alloc:
            print(f"\n  --- Dynamic Tensors ({total_alloc}) ---")
            for dt, cnt in alloc_dtypes.most_common():
                print(f"    {dt}: {cnt}")

        # Operator call trace
        if show_trace and plan.chains:
            instructions = plan.chains[0].instructions
            print(f"\n  --- Operator Call Trace ({len(instructions)} instructions) ---")
            for i, inst in enumerate(instructions):
                ka = inst.instr_args
                if not hasattr(ka, "op_index"):
                    continue
                op = op_names.get(ka.op_index, "?")
                args = list(ka.args) if hasattr(ka, "args") else []
                if trace_limit and i < trace_limit:
                    print(f"    [{i:4d}] {op}  args={args}")
                elif trace_limit and i == trace_limit:
                    remaining = len(instructions) - 2 * trace_limit
                    if remaining > 0:
                        print(f"    ... ({remaining} more instructions) ...")
                if not trace_limit or i >= len(instructions) - trace_limit:
                    if trace_limit and i >= trace_limit:
                        print(f"    [{i:4d}] {op}  args={args}")

    # ---- Segments ----
    if program.segments:
        print(f"\n=== SEGMENTS ===")
        for j, seg in enumerate(program.segments):
            print(f"  Segment {j}: offset={seg.offset}, size={seg.size}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pte_file", help="Path to .pte file")
    parser.add_argument(
        "--trace", action="store_true", help="Show full operator call trace"
    )
    parser.add_argument(
        "--trace-limit",
        type=int,
        default=80,
        metavar="N",
        help="Show first/last N instructions in trace (0 for all, default: 80)",
    )
    args = parser.parse_args()
    analyze(args.pte_file, show_trace=args.trace, trace_limit=args.trace_limit)


if __name__ == "__main__":
    main()
