import numpy as np
import torch
from torch import nn

import fpga_kernels


class FpgaReLU6(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:

        original_device = input_tensor.device

        input_numpy = np.ascontiguousarray(
            input_tensor.detach().cpu().numpy(),
            dtype=np.float32
        )

        output_numpy = fpga_kernels.relu6_forward(
            input_numpy
        )

        output_tensor = torch.from_numpy(
            output_numpy
        )

        return output_tensor.to(
            original_device
        )
