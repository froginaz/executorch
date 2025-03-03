# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Callable, Dict

import torch
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm.tosa_quant_utils import QuantArgs
from executorch.exir import ExportedProgram

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload

from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule

from torch.library import impl, Library

lib = Library("tosa", "DEF")
lib.define("_table(Tensor self) -> Tensor")


@impl(lib, "_table")
def _table_impl(*args, **kwargs):  # pyre-ignore
    in_dtype = args[0].dtype
    if in_dtype == torch.int8:
        return args[0]
    return args[0].to(dtype=torch.int32)


class InsertTableOpsPass(ExportPass):
    """
    For ops in self.table_ops they need to be serialized as a TOSA TABLE. This pass replaces these
    edge ops with a tosa._table(input: Tensor, target_str: str) where target_str == str(node.target).
    When lowering the _table node target_str will be used to find the corresponding torch operator
    which will be used to produce the table values in operators/op_table.py.
    """

    table_ops: Dict[EdgeOpOverload, Callable[[torch.Tensor], torch.Tensor]] = {
        exir_ops.edge.aten.exp.default: torch.exp,
        exir_ops.edge.aten.floor.default: torch.floor,
        exir_ops.edge.aten.log.default: torch.log,
        exir_ops.edge.aten.reciprocal.default: torch.reciprocal,
        exir_ops.edge.aten.rsqrt.default: torch.rsqrt,
        exir_ops.edge.aten.sigmoid.default: torch.sigmoid,
        exir_ops.edge.aten.tanh.default: torch.tanh,
        exir_ops.edge.aten.hardsigmoid.default: torch.nn.functional.hardsigmoid,
        exir_ops.edge.aten.hardswish.default: torch.nn.functional.hardswish,
    }

    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__()
        self.exported_program = exported_program

    def register_buffer(self, buffer_name: str, buffer: torch.Tensor) -> None:
        """
        Add buffer to self.exported_program.state_dict
        """
        self.exported_program.state_dict[buffer_name] = buffer

    def generate_8bit_table_values(
        self,
        torch_op: Callable[[torch.Tensor], torch.Tensor],
        in_quantargs: QuantArgs,
        out_quantargs: QuantArgs,
    ) -> tuple[torch.Tensor, int]:
        """Compute LUT values for a INT8 TOSA.TABLE. Also returns 0 since no shifting is required after 8bit table.
        The INT8 table is a simple 256 value 1-1 LUT.
        """

        def f(x: torch.Tensor) -> torch.Tensor:
            x = in_quantargs.dequantize_value(x)
            x = torch_op(x)
            return out_quantargs.quantize_value(x)

        return (
            f(
                torch.linspace(
                    start=in_quantargs.qmin,
                    end=in_quantargs.qmax,
                    steps=256,
                    # use torch.int64 to avoid overflow when dequantizing (subtracting zp).
                    # e.g. torch.tensor(-50, dtype=torch.int8) - 100 == torch.tensor(106, dtype=torch.int8)
                    dtype=torch.int64,
                )
            ).to(dtype=torch.int8),
            0,
        )

    def generate_16_bit_table_values(
        self,
        torch_op: Callable[[torch.Tensor], torch.Tensor],
        in_quantargs: QuantArgs,
        out_quantargs: QuantArgs,
    ) -> tuple[torch.Tensor, int]:
        """Compute LUT values for a INT16 TOSA.TABLE with 32 bit output (in practice 23 bit, see specification).
        The output of the the table will have 7 fractional bits, which means the output will interpreted as
        x128 times too large unless accounted for. Right shift the table values to fit
        in 16 bits. Return a lshift of the right shift - 7 due to the fractional bits.
        """

        def f(x: torch.Tensor) -> torch.Tensor:
            # Dont use the 7 LSBs
            x = in_quantargs.dequantize_value((x & ~0x7F))
            x = torch_op(x)
            return out_quantargs.quantize_value(x)

        lut_values = f(
            torch.linspace(
                start=in_quantargs.qmin,
                end=in_quantargs.qmax + 1,
                steps=513,
                # use torch.int64 to avoid overflow when dequantizing (subtracting zp).
                # e.g. torch.tensor(-50, dtype=torch.int8) - 100 == torch.tensor(106, dtype=torch.int8)
                dtype=torch.int64,
            )
        )
        # Calculate how much we need to shift table values to fit in 16 bits
        # ceil(log2(max absolute table value)) + 1 bit for signedness - 16
        # Note: for out_quantargs.dtype=torch.int16, rshift == 0.
        rshift = int(torch.ceil(torch.log2(lut_values.abs().max()))) + 1 - 16
        lut_values = lut_values >> rshift
        return lut_values.to(dtype=torch.int16), rshift - 7

    def generate_table_values(
        self,
        torch_op: Callable[[torch.Tensor], torch.Tensor],
        in_quantargs: QuantArgs,
        out_quantargs: QuantArgs,
    ) -> tuple[torch.Tensor, int]:
        match out_quantargs.dtype:
            case torch.int8:
                return self.generate_8bit_table_values(
                    torch_op, in_quantargs, out_quantargs
                )
            case torch.int16 | torch.int32:
                return self.generate_16_bit_table_values(
                    torch_op, in_quantargs, out_quantargs
                )
            case _:
                raise ValueError(
                    f"Unsupported output dtype for table: {out_quantargs.dtype}"
                )

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function" or node.target not in self.table_ops:
                continue
            input_qparams = node.meta["input_qparams"]
            output_qparams = node.meta["output_qparams"]
            if len(input_qparams) == 0 or len(output_qparams) == 0:
                # We only want to replace the node if it's quantized
                continue
            # Create table node
            with graph_module.graph.inserting_before(node):
                table_node = create_node(
                    graph=graph_module.graph,
                    op_target=torch.ops.tosa._table.default,
                    args=(node.args[0],),
                )
                output_node = table_node
                assert len(input_qparams) == 1
                assert len(output_qparams) == 1

                # Generate table buffer and how much to lshift the table output.
                buffer, lshift = self.generate_table_values(
                    torch_op=self.table_ops[node.target],
                    in_quantargs=input_qparams[0],
                    out_quantargs=output_qparams[0],
                )
                # Register buffer in self.exported_program.state_dict
                # When the graph is retraced, the implementation _table is used and the suffix _default disappears from the node name
                # Remove it here to make it possible to find in the node_visitor
                self.register_buffer(
                    buffer_name=table_node.name.replace("_default", ""), buffer=buffer
                )

                if lshift != 0:
                    scale = 2.0**lshift
                    rescale_node = create_node(
                        graph=graph_module.graph,
                        op_target=torch.ops.tosa._rescale.default,
                        args=(table_node, output_qparams[0].dtype, scale, 0, 0),
                    )
                    output_node = rescale_node

                node.replace_all_uses_with(output_node)
            graph_module.graph.erase_node(node)
            output_node.meta["input_qparams"] = input_qparams
            output_node.meta["output_qparams"] = output_qparams
            modified = True

        if modified:
            # retrace the graph to update the fake tensor types
            graph_module = super().call(graph_module).graph_module

            graph_module.recompile()
        return PassResult(graph_module, modified)
