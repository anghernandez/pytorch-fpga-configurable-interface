from .Tanh import FpgaTanh
from .Linear import FpgaLinear
from .Conv2d import FpgaConv2D
from .AvgPool2d import FpgaAvgPool2D

__all__ = [
    "FpgaTanh",
    "FpgaLinear",
    "FpgaConv2D",
    "FpgaAvgPool2D"
]
