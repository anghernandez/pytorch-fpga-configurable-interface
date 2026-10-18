import math

import numpy as np
import torch
from torch import nn

from fpga_graph import FpgaTensor


class FpgaDepthwiseConv2d(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False
    ) -> None:

        super().__init__()

        if channels <= 0:
            raise ValueError(
                "channels debe ser mayor que cero"
            )

        if kernel_size <= 0:
            raise ValueError(
                "kernel_size debe ser mayor que cero"
            )

        if stride <= 0:
            raise ValueError(
                "stride debe ser mayor que cero"
            )

        if padding < 0:
            raise ValueError(
                "padding no puede ser negativo"
            )

        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Forma compatible con PyTorch Depthwise Conv2d:
        # [channels, 1, kernel_height, kernel_width]
        self.weight = nn.Parameter(
            torch.empty(
                channels,
                1,
                kernel_size,
                kernel_size,
                dtype=torch.float32
            ),
            requires_grad=False
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    channels,
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

            fan_in = (
                self.kernel_size
                * self.kernel_size
            )

            bound = 1.0 / math.sqrt(
                fan_in
            )

            nn.init.uniform_(
                self.bias,
                -bound,
                bound
            )

    def forward(self, x: FpgaTensor) -> FpgaTensor:
        g = x.graph
        if len(x.shape) != 4 or x.shape[1] != self.channels:
            raise ValueError("Depthwise requiere NCHW con canales correctos")
        b, c, h, w = x.shape
        kh = kw = self.kernel_size
        sh = sw = self.stride
        ph = pw = self.padding
        oh, ow = (h+2*ph-kh)//sh+1, (w+2*pw-kw)//sw+1
        if min(oh,ow) <= 0: raise ValueError("Salida Depthwise inválida")
        params = g.register_weight_bias(self, "depthwise_conv2d_forward", 1, 2, c)
        return g.add_node("Depthwise", [x], params, (b,c,oh,ow), {
            "batch_size": b, "channels": c, "input_height": h, "input_width": w,
            "output_height": oh, "output_width": ow,
            "kernel_height": kh, "kernel_width": kw,
            "stride_height": sh, "stride_width": sw,
            "padding_height": ph, "padding_width": pw,
            "use_bias": int(self.bias is not None)})