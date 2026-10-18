import math

import numpy as np
import torch
from torch import nn

from fpga_graph import FpgaTensor


class FpgaConv2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        bias: bool = True
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels debe ser mayor que cero"
            )

        if out_channels <= 0:
            raise ValueError(
                "out_channels debe ser mayor que cero"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = self._to_pair(
            kernel_size,
            "kernel_size"
        )

        self.stride = self._to_pair(
            stride,
            "stride"
        )

        self.padding = self._to_pair(
            padding,
            "padding"
        )

        if (
            self.kernel_size[0] <= 0 or
            self.kernel_size[1] <= 0
        ):
            raise ValueError(
                "kernel_size debe ser mayor que cero"
            )

        if (
            self.stride[0] <= 0 or
            self.stride[1] <= 0
        ):
            raise ValueError(
                "stride debe ser mayor que cero"
            )

        if (
            self.padding[0] < 0 or
            self.padding[1] < 0
        ):
            raise ValueError(
                "padding no puede ser negativo"
            )

        kernel_height = self.kernel_size[0]
        kernel_width = self.kernel_size[1]

        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                kernel_height,
                kernel_width,
                dtype=torch.float32
            ),
            requires_grad=False
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    out_channels,
                    dtype=torch.float32
                ),
                requires_grad=False
            )
        else:
            self.register_parameter(
                "bias",
                None
            )

        self.reset_parameters()

    @staticmethod
    def _to_pair(
        value,
        name: str
    ):
        if isinstance(value, int):
            return (value, value)

        if (
            isinstance(value, tuple) and
            len(value) == 2
        ):
            return value

        raise ValueError(
            f"{name} debe ser un entero o una tupla de dos enteros"
        )

    def reset_parameters(self) -> None:
        kernel_height = self.kernel_size[0]
        kernel_width = self.kernel_size[1]

        fan_in = (
            self.in_channels
            * kernel_height
            * kernel_width
        )

        bound = 1.0 / math.sqrt(
            fan_in
        )

        with torch.no_grad():
            self.weight.uniform_(
                -bound,
                bound
            )

            if self.bias is not None:
                self.bias.uniform_(
                    -bound,
                    bound
                )

    def forward(self, x: FpgaTensor) -> FpgaTensor:
        g = x.graph
        if len(x.shape) != 4 or x.shape[1] != self.in_channels:
            raise ValueError("Conv2D requiere [batch, in_channels, height, width]")
        b, _, h, w = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        oh, ow = (h + 2*ph - kh)//sh + 1, (w + 2*pw - kw)//sw + 1
        if min(oh, ow) <= 0: raise ValueError("Salida Conv2D inválida")
        params = g.register_weight_bias(self, "conv2d_forward", 1, 2, self.out_channels)
        return g.add_node("Conv2D", [x], params, (b, self.out_channels, oh, ow),
            {"batch_size": b, "input_channels": self.in_channels,
             "input_height": h, "input_width": w, "output_channels": self.out_channels,
             "output_height": oh, "output_width": ow,
             "kernel_height": kh, "kernel_width": kw,
             "stride_height": sh, "stride_width": sw,
             "padding_height": ph, "padding_width": pw,
             "use_bias": int(self.bias is not None)})