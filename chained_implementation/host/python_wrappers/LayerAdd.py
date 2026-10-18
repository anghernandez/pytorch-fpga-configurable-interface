import numpy as np
import torch
from torch import nn

from fpga_graph import FpgaTensor


class FpgaLayerAdd(nn.Module):

    def __init__(
        self,
        alpha: float = 1.0
    ) -> None:

        super().__init__()

        self.alpha = alpha

    def forward(self, input1: FpgaTensor, input2: FpgaTensor) -> FpgaTensor:
        if input1.graph is not input2.graph or input1.shape != input2.shape:
            raise ValueError("LayerAdd requiere entradas del mismo grafo y forma")
        g = input1.graph
        return g.add_node("LayerAdd", [input1, input2], [], input1.shape,
            {"size": g.element_count(input1.shape)}, {"alpha": float(self.alpha)})