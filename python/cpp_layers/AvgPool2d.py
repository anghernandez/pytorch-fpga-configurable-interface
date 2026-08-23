import math

import numpy as np
import torch
from torch import nn

import cpp_kernels


def _to_pair(
    value: int | tuple[int, int],
    name: str
) -> tuple[int, int]:

    if isinstance(value, int):
        return value, value

    if isinstance(value, tuple) and len(value) == 2:
        return value

    raise ValueError(
        f"{name} debe ser un entero "
        "o una tupla de dos enteros"
    )


class CppAvgPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None
    ) -> None:
        super().__init__()

        self.kernel_size = _to_pair(
            kernel_size,
            "kernel_size"
        )

        if stride is None:
            stride = kernel_size

        self.stride = _to_pair(
            stride,
            "stride"
        )

        self.padding = _to_pair(
            padding,
            "padding"
        )

        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

        if min(self.kernel_size) <= 0:
            raise ValueError(
                "kernel_size debe ser mayor que cero"
            )

        if min(self.stride) <= 0:
            raise ValueError(
                "stride debe ser mayor que cero"
            )

        if min(self.padding) < 0:
            raise ValueError(
                "padding no puede ser negativo"
            )

        if (
            divisor_override is not None
            and divisor_override <= 0
        ):
            raise ValueError(
                "divisor_override debe ser "
                "mayor que cero"
            )

    def _output_size(
        self,
        input_size: int,
        kernel_size: int,
        stride: int,
        padding: int
    ) -> int:

        numerator = (
            input_size
            + 2 * padding
            - kernel_size
        )

        if self.ceil_mode:
            output_size = (
                math.ceil(numerator / stride)
                + 1
            )

            if (
                (output_size - 1) * stride
                >= input_size + padding
            ):
                output_size -= 1
        else:
            output_size = (
                math.floor(numerator / stride)
                + 1
            )

        return output_size

    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:

        if input_tensor.ndim != 4:
            raise ValueError(
                "input debe tener forma "
                "[batch, channels, height, width]"
            )

        input_height = input_tensor.shape[2]
        input_width = input_tensor.shape[3]

        kernel_height, kernel_width = (
            self.kernel_size
        )

        stride_height, stride_width = (
            self.stride
        )

        padding_height, padding_width = (
            self.padding
        )

        output_height = self._output_size(
            input_height,
            kernel_height,
            stride_height,
            padding_height
        )

        output_width = self._output_size(
            input_width,
            kernel_width,
            stride_width,
            padding_width
        )

        if output_height <= 0 or output_width <= 0:
            raise ValueError(
                "La configuración genera una "
                "salida con dimensiones inválidas"
            )

        original_device = input_tensor.device

        input_numpy = np.ascontiguousarray(
            input_tensor.detach().cpu().numpy(),
            dtype=np.float32
        )

        output_numpy = (
            cpp_kernels.avg_pool2d_forward(
                input_numpy,
                output_height,
                output_width,
                kernel_height,
                kernel_width,
                stride_height,
                stride_width,
                padding_height,
                padding_width,
                self.count_include_pad,
                self.divisor_override
            )
        )

        return torch.from_numpy(
            output_numpy
        ).to(original_device)

