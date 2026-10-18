import numpy as np
import torch
from torch import nn

from fpga_graph import FpgaTensor

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

    def forward(self, x: FpgaTensor) -> FpgaTensor:
        g = x.graph
        if len(x.shape) != 4: raise ValueError("AvgPool2D requiere NCHW")
        b, c, h, w = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        # PyTorch AvgPool2d utiliza ceil_mode=False por defecto.
        oh, ow = (h + 2*ph - kh)//sh + 1, (w + 2*pw - kw)//sw + 1
        if min(oh, ow) <= 0: raise ValueError("Salida AvgPool2D inválida")
        return g.add_node("AvgPool2D", [x], [], (b,c,oh,ow), {
            "batch_size": b, "input_channels": c,
            "input_height": h, "input_width": w,
            "output_height": oh, "output_width": ow,
            "kernel_height": kh, "kernel_width": kw,
            "stride_height": sh, "stride_width": sw,
            "padding_height": ph, "padding_width": pw,
            "count_include_pad": int(self.count_include_pad),
            "use_divisor_override": int(self.divisor_override is not None),
            "divisor_override": int(self.divisor_override or 0)})