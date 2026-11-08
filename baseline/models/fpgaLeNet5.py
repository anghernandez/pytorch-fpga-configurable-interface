#VERSION SIN ENCADENAMIENTO
import torch
from torch import nn

# Wrappers Python de las capas ejecutadas en FPGA
from python_wrappers.Conv2d import FpgaConv2D
from python_wrappers.AvgPool2d import FpgaAvgPool2D
from python_wrappers.Linear import FpgaLinear
from python_wrappers.Tanh import FpgaTanh

# Red LeNet-5 ejecutada utilizando capas FPGA
class FpgaLeNet5(nn.Module):

    # Inicializa la arquitectura de la red
    def __init__(self) -> None:
        super().__init__()

        # Primera etapa convolucional
        self.conv1 = FpgaConv2D(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True
        )

        # Primera función de activación
        self.tanh1 = FpgaTanh()

        # Primera capa de Average Pooling
        self.pool1 = FpgaAvgPool2D(
            kernel_size=2,
            stride=2
        )

        # Segunda etapa convolucional
        self.conv2 = FpgaConv2D(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=True
        )

        # Segunda función de activación
        self.tanh2 = FpgaTanh()

        # Segunda capa de Average Pooling
        self.pool2 = FpgaAvgPool2D(
            kernel_size=2,
            stride=2
        )

        # Primera capa totalmente conectada
        self.fc1 = FpgaLinear(
            in_features=16 * 5 * 5,
            out_features=120,
            bias=True
        )

        # Tercera función de activación
        self.tanh3 = FpgaTanh()

        # Segunda capa totalmente conectada
        self.fc2 = FpgaLinear(
            in_features=120,
            out_features=84,
            bias=True
        )

        # Cuarta función de activación
        self.tanh4 = FpgaTanh()

        # Capa de salida
        self.fc3 = FpgaLinear(
            in_features=84,
            out_features=10,
            bias=True
        )

    # Ejecuta la inferencia de la red
    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:

        # Ejecuta la primera convolución
        output_tensor = self.conv1(
            input_tensor
        )

        # Aplica la primera activación
        output_tensor = self.tanh1(
            output_tensor
        )

        # Ejecuta el primer Average Pooling
        output_tensor = self.pool1(
            output_tensor
        )
        # Ejecuta la segunda convolución
        output_tensor = self.conv2(
            output_tensor
        )

        # Aplica la segunda activación
        output_tensor = self.tanh2(
            output_tensor
        )

        # Ejecuta el segundo Averege Pooling
        output_tensor = self.pool2(
            output_tensor
        )

        # Obtiene el tamaño del batch
        batch_size = output_tensor.shape[0]

        # Convierte la salida 4D en una matriz 2D
        # para alimentar la capa totalmente conectada
        output_tensor = output_tensor.reshape(
            batch_size,
            16 * 5 * 5
        )

        # Ejecuta la primera capa totalmente conectada
        output_tensor = self.fc1(
            output_tensor
        )

        # Aplica la tercera activación
        output_tensor = self.tanh3(
            output_tensor
        )

        #Ejecuta la segunda capa totalmente conectada
        output_tensor = self.fc2(
            output_tensor
        )

        # Aplica la cuarta activación
        output_tensor = self.tanh4(
            output_tensor
        )

        # Ejecuta la capa de salida
        output_tensor = self.fc3(
            output_tensor
        )

        # Devuelve las predicciones
        return output_tensor

    # Carga los pesos entrenados desde Pytorch
    def load_pytorch_weights(
        self,
        weights_path: str
    ) -> None:

        # Carga el archivo de pesos
        state_dict = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True
        )

        # Desactiva el cálculo de gradientes
        with torch.no_grad():

            # Carga pesos y bias de la primera convolución
            self.conv1.weight.copy_(
                state_dict["features.0.weight"]
            )

            self.conv1.bias.copy_(
                state_dict["features.0.bias"]
            )

            # Carga pesos y bias de la segunda convolución
            self.conv2.weight.copy_(
                state_dict["features.3.weight"]
            )

            self.conv2.bias.copy_(
                state_dict["features.3.bias"]
            )

            # Carga pesos y bias de la primera capa lineal
            self.fc1.weight.copy_(
                state_dict["classifier.1.weight"]
            )

            self.fc1.bias.copy_(
                state_dict["classifier.1.bias"]
            )

            # Carga pesos y bias de la segunda capa lineal
            self.fc2.weight.copy_(
                state_dict["classifier.3.weight"]
            )

            self.fc2.bias.copy_(
                state_dict["classifier.3.bias"]
            )

            # Carga pesos y bias de la capa de salida
            self.fc3.weight.copy_(
                state_dict["classifier.5.weight"]
            )

            self.fc3.bias.copy_(
                state_dict["classifier.5.bias"]
            )

        # Confirma que los pesos fueron cargados
        print(
            "Pesos de PyTorch cargados "
            "correctamente en FpgaLeNet5."
        )