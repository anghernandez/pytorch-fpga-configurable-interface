import numpy as np
import torch
from torch import nn

from fpga_graph import FpgaTensor


class FpgaBatchNorm2d(nn.Module):

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        bias: bool = True
    ) -> None:

        super().__init__()

        if num_features <= 0:
            raise ValueError(
                "num_features debe ser mayor que cero"
            )

        if eps <= 0:
            raise ValueError(
                "eps debe ser mayor que cero"
            )

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:

            self.weight = nn.Parameter(
                torch.ones(
                    num_features,
                    dtype=torch.float32
                ),
                requires_grad=False
            )

            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(
                        num_features,
                        dtype=torch.float32
                    ),
                    requires_grad=False
                )
            else:
                self.register_parameter(
                    "bias",
                    None
                )

        else:
            self.register_parameter(
                "weight",
                None
            )

            self.register_parameter(
                "bias",
                None
            )

        self.register_buffer(
            "running_mean",
            torch.zeros(
                num_features,
                dtype=torch.float32
            )
        )

        self.register_buffer(
            "running_var",
            torch.ones(
                num_features,
                dtype=torch.float32
            )
        )

    def forward(self, x: FpgaTensor) -> FpgaTensor:
        g = x.graph
        if self.training:
            raise RuntimeError("BatchNorm2D FPGA solo admite inferencia: use model.eval()")
        if len(x.shape) != 4 or x.shape[1] != self.num_features:
            raise ValueError("BatchNorm2D requiere NCHW con canales correctos")
        b,c,h,w = x.shape
        import torch
        weight = self.weight if self.affine else torch.ones(c, dtype=torch.float32)
        bias = self.bias if self.bias is not None else torch.zeros(c, dtype=torch.float32)
        params = [g.register_parameter(self, "bn_weight", weight, "batchnorm2d_forward", 1),
                  g.register_parameter(self, "bn_bias", bias, "batchnorm2d_forward", 2),
                  g.register_parameter(self, "bn_mean", self.running_mean, "batchnorm2d_forward", 3),
                  g.register_parameter(self, "bn_var", self.running_var, "batchnorm2d_forward", 4)]
        return g.add_node("BatchNorm2D", [x], params, x.shape, {
            "batch_size": b, "channels": c,
            "input_height": h, "input_width": w,
            "affine": int(self.affine), "use_bias": int(self.bias is not None)},
            {"eps": float(self.eps)})