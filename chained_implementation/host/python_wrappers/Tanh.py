"""Tanh diferida: registra un nodo para ejecutarlo posteriormente en FPGA."""
from torch import nn
from fpga_graph import FpgaTensor


class FpgaTanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor: FpgaTensor) -> FpgaTensor:
        if not isinstance(input_tensor, FpgaTensor):
            raise TypeError("FpgaTanh requiere FpgaTensor; construya el DAG con FpgaGraph.build()")
        return input_tensor.graph.add_unary("Tanh", input_tensor)