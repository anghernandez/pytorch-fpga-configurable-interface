import math

import numpy as np
import torch
from torch import nn

from fpga_graph import FpgaTensor


class FpgaPointwiseConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False
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

        # Forma compatible con PyTorch Conv2d 1x1:
        # [out_channels, in_channels, 1, 1]
        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                1,
                1,
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

    def reset_parameters(self) -> None:

        nn.init.kaiming_uniform_(
            self.weight,
            a=math.sqrt(5)
        )

        if self.bias is not None:

            bound = 1.0 / math.sqrt(
                self.in_channels
            )

            nn.init.uniform_(
                self.bias,
                -bound,
                bound
            )

    def forward(self, x: FpgaTensor) -> FpgaTensor:
        g = x.graph
        if len(x.shape) != 4 or x.shape[1] != self.in_channels:
            raise ValueError("Pointwise requiere NCHW con canales correctos")
        b, _, h, w = x.shape
        params = g.register_weight_bias(self, "pointwise_conv2d_forward", 1, 2, self.out_channels)
        return g.add_node("Pointwise", [x], params, (b,self.out_channels,h,w), {
            "batch_size": b, "input_channels": self.in_channels,
            "input_height": h, "input_width": w,
            "output_channels": self.out_channels,
            "use_bias": int(self.bias is not None)})