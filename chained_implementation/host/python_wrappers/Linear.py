import math

import numpy as np
import torch
from torch import nn

from fpga_graph import FpgaTensor


class FpgaLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ) -> None:
        super().__init__()

        if in_features <= 0:
            raise ValueError(
                "in_features debe ser mayor que cero"
            )

        if out_features <= 0:
            raise ValueError(
                "out_features debe ser mayor que cero"
            )

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                dtype=torch.float32
            ),
            requires_grad=False
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    out_features,
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
        bound = 1.0 / math.sqrt(
            self.in_features
        )

        with torch.no_grad():
            self.weight.uniform_(
                -bound,
                bound
            )

            if self.bias is not None:
                self.bias.uniform_(
                    -bound,
                    bound
                )

    def forward(self, x: FpgaTensor) -> FpgaTensor:
        g = x.graph
        if len(x.shape) != 2 or x.shape[1] != self.in_features:
            raise ValueError("Linear requiere [batch, in_features]")
        params = g.register_weight_bias(self, "linear_forward", 1, 2, self.out_features)
        return g.add_node("Linear", [x], params, (x.shape[0], self.out_features),
            {"batch_size": x.shape[0], "in_features": self.in_features,
             "out_features": self.out_features, "use_bias": int(self.bias is not None)})