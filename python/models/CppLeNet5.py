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

        # Primera etapa convolucional
        self.conv1 = CppConv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True
        )

        self.tanh1 = CppTanh()

        self.pool1 = CppAvgPool2d(
            kernel_size=2,
            stride=2
        )

        # Segunda etapa convolucional
        self.conv2 = CppConv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=True
        )

        self.tanh2 = CppTanh()

        self.pool2 = CppAvgPool2d(
            kernel_size=2,
            stride=2
        )

        # Capas densas
        self.fc1 = CppLinear(
            in_features=16 * 5 * 5,
            out_features=120,
            bias=True
        )

        self.tanh3 = CppTanh()

        self.fc2 = CppLinear(
            in_features=120,
            out_features=84,
            bias=True
        )

        self.tanh4 = CppTanh()

        self.fc3 = CppLinear(
            in_features=84,
            out_features=10,
            bias=True
        )

    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:

        output_tensor = self.conv1(
            input_tensor
        )

        output_tensor = self.tanh1(
            output_tensor
        )

        output_tensor = self.pool1(
            output_tensor
        )

        output_tensor = self.conv2(
            output_tensor
        )

        output_tensor = self.tanh2(
            output_tensor
        )

        output_tensor = self.pool2(
            output_tensor
        )

        batch_size = output_tensor.shape[0]

        output_tensor = output_tensor.reshape(
            batch_size,
            16 * 5 * 5
        )

        output_tensor = self.fc1(
            output_tensor
        )

        output_tensor = self.tanh3(
            output_tensor
        )

        output_tensor = self.fc2(
            output_tensor
        )

        output_tensor = self.tanh4(
            output_tensor
        )

        output_tensor = self.fc3(
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

        with torch.no_grad():

            self.conv1.weight.copy_(
                state_dict["features.0.weight"]
            )

            self.conv1.bias.copy_(
                state_dict["features.0.bias"]
            )

            self.conv2.weight.copy_(
                state_dict["features.3.weight"]
            )

            self.conv2.bias.copy_(
                state_dict["features.3.bias"]
            )

            self.fc1.weight.copy_(
                state_dict["classifier.1.weight"]
            )

            self.fc1.bias.copy_(
                state_dict["classifier.1.bias"]
            )

            self.fc2.weight.copy_(
                state_dict["classifier.3.weight"]
            )

            self.fc2.bias.copy_(
                state_dict["classifier.3.bias"]
            )

            self.fc3.weight.copy_(
                state_dict["classifier.5.weight"]
            )

            self.fc3.bias.copy_(
                state_dict["classifier.5.bias"]
            )

        print("PyTorch weights loaded successfully.")