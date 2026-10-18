"""ReLU6 diferida: registra un nodo para ejecutarlo posteriormente en FPGA."""
from torch import nn
from fpga_graph import FpgaTensor


class FpgaReLU6(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor: FpgaTensor) -> FpgaTensor:
        if not isinstance(input_tensor, FpgaTensor):
            raise TypeError("FpgaReLU6 requiere FpgaTensor; construya el DAG con FpgaGraph.build()")
        return input_tensor.graph.add_unary("ReLU6", input_tensor)