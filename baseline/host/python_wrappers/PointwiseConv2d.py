import math

import numpy as np
import torch
from torch import nn

import fpga_kernels


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

    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:

        if input_tensor.ndim != 4:
            raise ValueError(
                "input debe tener forma "
                "[batch_size, channels, height, width]"
            )

        if input_tensor.shape[1] != self.in_channels:
            raise ValueError(
                "Los canales de entrada no coinciden "
                "con in_channels"
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

        output_numpy = fpga_kernels.pointwise_forward(
            input_numpy,
            weight_numpy,
            bias_numpy,
            use_bias
        )

        output_tensor = torch.from_numpy(
            output_numpy
        )

        return output_tensor.to(
            original_device
        )


"""
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

        # ========================================================
        # Parámetros PyTorch
        # ========================================================

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

        # ========================================================
        # Parámetros residentes en FPGA
        #
        # Se inicializan como None.
        # Se cargan en el primer forward().
        # ========================================================

        self._weight_fpga = None
        self._bias_fpga = None

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

    def _load_parameters_to_fpga(self) -> None:

        # ========================================================
        # Weight
        # ========================================================

        weight_numpy = np.ascontiguousarray(
            self.weight.detach().cpu().numpy(),
            dtype=np.float32
        )

        self._weight_fpga = (
            fpga_buffer.pointwise_weight_from_numpy(
                weight_numpy
            )
        )

        # ========================================================
        # Bias
        # ========================================================

        if self.bias is not None:

            bias_numpy = np.ascontiguousarray(
                self.bias.detach().cpu().numpy(),
                dtype=np.float32
            )

            self._bias_fpga = (
                fpga_buffer.pointwise_bias_from_numpy(
                    bias_numpy
                )
            )

        else:

            # El kernel necesita un buffer de bias aunque
            # use_bias sea False.
            bias_numpy = np.zeros(
                self.out_channels,
                dtype=np.float32
            )

            self._bias_fpga = (
                fpga_buffer.pointwise_bias_from_numpy(
                    bias_numpy
                )
            )

    def forward(self, input_tensor):

        # ========================================================
        # Validación de entrada
        # ========================================================

        if isinstance(input_tensor, torch.Tensor):

            if input_tensor.ndim != 4:
                raise ValueError(
                    "input debe tener forma "
                    "[batch_size, channels, height, width]"
                )

            if input_tensor.shape[1] != self.in_channels:
                raise ValueError(
                    "Los canales de entrada no coinciden "
                    "con in_channels"
                )

        elif isinstance(
            input_tensor,
            fpga_buffer.FpgaTensor
        ):

            if len(input_tensor.shape) != 4:
                raise ValueError(
                    "input FPGA debe tener forma "
                    "[batch_size, channels, height, width]"
                )

            if input_tensor.shape[1] != self.in_channels:
                raise ValueError(
                    "Los canales de entrada no coinciden "
                    "con in_channels"
                )

        else:

            raise TypeError(
                "input_tensor debe ser torch.Tensor "
                "o fpga_buffer.FpgaTensor"
            )

        # ========================================================
        # Cargar parámetros una sola vez
        # ========================================================

        if self._weight_fpga is None:

            self._load_parameters_to_fpga()

        # ========================================================
        # Caso 1:
        # Entrada ya residente en FPGA.
        #
        # No hay CPU -> FPGA para la activación.
        # ========================================================

        if isinstance(
            input_tensor,
            fpga_buffer.FpgaTensor
        ):

            return fpga_buffer.pointwise_forward(
                input_tensor,
                self._weight_fpga,
                self._bias_fpga,
                self.bias is not None
            )

        # ========================================================
        # Caso 2:
        # Entrada normal de PyTorch.
        #
        # Esta es la entrada inicial al pipeline FPGA.
        # ========================================================

        original_device = input_tensor.device

        input_numpy = np.ascontiguousarray(
            input_tensor.detach().cpu().numpy(),
            dtype=np.float32
        )

        input_fpga = fpga_buffer.from_numpy(
            input_numpy
        )

        output_fpga = fpga_buffer.pointwise_forward(
            input_fpga,
            self._weight_fpga,
            self._bias_fpga,
            self.bias is not None
        )

        # ========================================================
        # Compatibilidad con PyTorch:
        # solamente aquí regresamos a CPU.
        # ========================================================

        output_numpy = output_fpga.to_numpy()

        output_tensor = torch.from_numpy(
            output_numpy
        )

        return output_tensor.to(
            original_device
        )

"""