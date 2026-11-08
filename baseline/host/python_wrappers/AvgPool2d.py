import numpy as np
import torch
from torch import nn

import fpga_kernels

# Módulo de AvgPool2D ejecutado en la FPGA
class FpgaAvgPool2D(nn.Module):

    # Iniciación de los parametros de la capa
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        count_include_pad: bool = True,
        divisor_override=None
    ) -> None:
        super().__init__()

        # Convierte el kernel_size en una tupla (alto, ancho)
        self.kernel_size = self._to_pair(
            kernel_size,
            "kernel_size"
        )

        # Si no se especifica stride, utiliza kernel_size.
        if stride is None:
            self.stride = self.kernel_size
        else:
            # Convierte stride a una tupla (alto, ancho)
            self.stride = self._to_pair(
                stride,
                "stride"
            )
        #Convierte padding a una tupla (alto, ancho).
        self.padding = self._to_pair(
            padding,
            "padding"
        )

        #  Verfica que el tamaño del kernel sea válido.
        if (
            self.kernel_size[0] <= 0 or
            self.kernel_size[1] <= 0
        ):
            raise ValueError(
                "kernel_size debe ser mayor que cero"
            )

        # Verifica que el stride sea válido
        if (
            self.stride[0] <= 0 or
            self.stride[1] <= 0
        ):
            raise ValueError(
                "stride debe ser mayor que cero"
            )
        
        # Verifica que el padding no sea negativo
        if (
            self.padding[0] < 0 or
            self.padding[1] < 0
        ):
            raise ValueError(
                "padding no puede ser negativo"
            )

        # Verifica que el divisor_override sea válido
        if (
            divisor_override is not None and
            divisor_override <= 0
        ):
            raise ValueError(
                "divisor_override debe ser mayor que cero"
            )

        # Guarda si se incluyen los elementos del padding
        self.count_include_pad = bool(
            count_include_pad
        )

        # Guarda el divisor personalizado
        self.divisor_override = divisor_override

    # Convierte un entero o una tupla en (alto, ancho)
    @staticmethod
    def _to_pair(
        value,
        name: str
    ):  
        # Convierte un entero en una tupla
        if isinstance(value, int):
            return (value, value)
        
        # Verifica que sea una tupla de dos enteros
        if (
            isinstance(value, tuple) and
            len(value) == 2 and
            isinstance(value[0], int) and
            isinstance(value[1], int)
        ):
            return value

        # Rechaza formatos no válidos
        raise ValueError(
            f"{name} debe ser un entero "
            "o una tupla de dos enteros"
        )

    # Ejecuta la capa de AvgPool2d
    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:

        # Verifica que la entrada sea de cuatro dimensiones
        if input_tensor.ndim != 4:
            raise ValueError(
                "input debe tener forma "
                "[batch_size, channels, height, width]"
            )

        # Verifica que el batch_size sea válido
        if input_tensor.shape[0] <= 0:
            raise ValueError(
                "batch_size debe ser mayor que cero"
            )

        # Verifica que el número de canales sea válido
        if input_tensor.shape[1] <= 0:
            raise ValueError(
                "channels debe ser mayor que cero"
            )

        # Verifica que heigth y width sean válidos
        if (
            input_tensor.shape[2] <= 0 or
            input_tensor.shape[3] <= 0
        ):
            raise ValueError(
                "height y width deben ser mayores que cero"
            )

        # Guarda el dispositivo original del tensor
        original_device = input_tensor.device

        # Convierte el tensor de Pytorch a Numpy
        # y garantiza memoria contigua en formato float32
        input_numpy = np.ascontiguousarray(
            input_tensor.detach().cpu().numpy(), # esto regresa a la CPU
            dtype=np.float32
        )

        # Configura el uso de divisor_override
        if self.divisor_override is None:
            use_divisor_override = False
            divisor_override = 0
        else:
            use_divisor_override = True
            divisor_override = int(
                self.divisor_override
            )
        # Llama a la función de C++ expuesta mediante Pybind11
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

        # Convierte la salida Numpy nuevamente a Tensor
        output_tensor = torch.from_numpy(
            output_numpy
        )

        # Devuelve la salida al dispositivo original
        return output_tensor.to(
            original_device
        )