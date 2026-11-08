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


"""
import numpy as np
import torch
from torch import nn

import fpga_buffer


class FpgaReLU6(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor):

        # ========================================================
        # Caso 1:
        # La entrada ya está en FPGA.
        #
        # No hacemos ninguna transferencia FPGA -> CPU.
        # ========================================================

        if isinstance(input_tensor, fpga_buffer.FpgaTensor):

            return fpga_buffer.relu6_forward(
                input_tensor
            )

        # ========================================================
        # Caso 2:
        # La entrada todavía está en PyTorch/CPU.
        #
        # Se realiza la transferencia inicial CPU -> FPGA.
        # ========================================================

        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError(
                "input_tensor debe ser torch.Tensor "
                "o fpga_buffer.FpgaTensor"
            )

        original_device = input_tensor.device

        input_numpy = np.ascontiguousarray(
            input_tensor.detach().cpu().numpy(),
            dtype=np.float32
        )

        input_fpga = fpga_buffer.from_numpy(
            input_numpy
        )

        output_fpga = fpga_buffer.relu6_forward(
            input_fpga
        )

        # ========================================================
        # Compatibilidad con la interfaz PyTorch actual.
        #
        # Solo aquí se realiza FPGA -> CPU.
        # ========================================================

        output_numpy = output_fpga.to_numpy()

        output_tensor = torch.from_numpy(
            output_numpy
        )

        return output_tensor.to(
            original_device
        )

"""