import math

import numpy as np
import torch
from torch import nn

import fpga_kernels


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

    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:

        if input_tensor.ndim != 4:
            raise ValueError(
                "input debe tener forma "
                "[batch_size, in_channels, height, width]"
            )

        if input_tensor.shape[1] != self.in_channels:
            raise ValueError(
                "input.shape[1] no coincide con in_channels"
            )

        original_device = input_tensor.device

        input_numpy = np.ascontiguousarray(
            input_tensor.detach().cpu().numpy(),
            dtype=np.float32
        )

        weight_numpy = np.ascontiguousarray(
            self.weight.detach().cpu().numpy(),
            dtype=np.float32
        )

        if self.bias is not None:
            bias_numpy = np.ascontiguousarray(
                self.bias.detach().cpu().numpy(),
                dtype=np.float32
            )
            use_bias = True
        else:
            bias_numpy = np.zeros(
                self.out_channels,
                dtype=np.float32
            )
            use_bias = False

        output_numpy = fpga_kernels.conv2d_forward(
            input_numpy,
            weight_numpy,
            bias_numpy,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            use_bias
        )

        output_tensor = torch.from_numpy(
            output_numpy
        )

        return output_tensor.to(
            original_device
        )

