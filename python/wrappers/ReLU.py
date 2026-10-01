import numpy as np
import torch
from torch import nn

import cpp_kernels


class CppReLU(nn.Module):
    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:

        original_device = input_tensor.device

        input_numpy = np.ascontiguousarray(
            input_tensor.detach().cpu().numpy(),
            dtype=np.float32
        )

        output_numpy = cpp_kernels.relu_forward(
            input_numpy
        )

        output_tensor = torch.from_numpy(
            output_numpy
        )

        return output_tensor.to(original_device)