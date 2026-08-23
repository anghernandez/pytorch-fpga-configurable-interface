import torch
from torch import nn

from cpp_layers import (
    CppAvgPool2d,
    CppConv2d,
    CppLinear,
    CppTanh,
)


class CppLeNet5(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            CppConv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=True
            ),
            CppTanh(),
            CppAvgPool2d(
                kernel_size=2,
                stride=2
            ),

            CppConv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0,
                bias=True
            ),
            CppTanh(),
            CppAvgPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            CppLinear(
                in_features=16 * 5 * 5,
                out_features=120,
                bias=True
            ),
            CppTanh(),

            CppLinear(
                in_features=120,
                out_features=84,
                bias=True
            ),
            CppTanh(),

            CppLinear(
                in_features=84,
                out_features=10,
                bias=True
            )
        )

    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:

        output_tensor = self.features(
            input_tensor
        )

        output_tensor = self.classifier(
            output_tensor
        )

        return output_tensor

    def load_pytorch_weights(
        self,
        weights_path: str
    ) -> None:

        state_dict = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True
        )

        self.load_state_dict(
            state_dict
        )
