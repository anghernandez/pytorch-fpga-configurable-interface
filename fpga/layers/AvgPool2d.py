import numpy as np
import torch
from torch import nn

import fpga_kernels


class FpgaAvgPool2D(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        count_include_pad: bool = True,
        divisor_override=None
    ) -> None:
        super().__init__()

        self.kernel_size = self._to_pair(
            kernel_size,
            "kernel_size"
        )

        if stride is None:
            self.stride = self.kernel_size
        else:
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

        if (
            divisor_override is not None and
            divisor_override <= 0
        ):
            raise ValueError(
                "divisor_override debe ser mayor que cero"
            )

        self.count_include_pad = bool(
            count_include_pad
        )

        self.divisor_override = divisor_override

    @staticmethod
    def _to_pair(
        value,
        name: str
    ):
        if isinstance(value, int):
            return (value, value)

        if (
            isinstance(value, tuple) and
            len(value) == 2 and
            isinstance(value[0], int) and
            isinstance(value[1], int)
        ):
            return value

        raise ValueError(
            f"{name} debe ser un entero "
            "o una tupla de dos enteros"
        )

    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:

        if input_tensor.ndim != 4:
            raise ValueError(
                "input debe tener forma "
                "[batch_size, channels, height, width]"
            )

        if input_tensor.shape[0] <= 0:
            raise ValueError(
                "batch_size debe ser mayor que cero"
            )

        if input_tensor.shape[1] <= 0:
            raise ValueError(
                "channels debe ser mayor que cero"
            )

        if (
            input_tensor.shape[2] <= 0 or
            input_tensor.shape[3] <= 0
        ):
            raise ValueError(
                "height y width deben ser mayores que cero"
            )

        original_device = input_tensor.device

        input_numpy = np.ascontiguousarray(
            input_tensor.detach().cpu().numpy(),
            dtype=np.float32
        )

        if self.divisor_override is None:
            use_divisor_override = False
            divisor_override = 0
        else:
            use_divisor_override = True
            divisor_override = int(
                self.divisor_override
            )

        output_numpy = fpga_kernels.avgpool2d_forward(
            input_numpy,
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.count_include_pad,
            use_divisor_override,
            divisor_override
        )

        output_tensor = torch.from_numpy(
            output_numpy
        )

        return output_tensor.to(
            original_device
        )