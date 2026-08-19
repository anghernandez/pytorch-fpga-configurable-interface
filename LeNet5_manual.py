import math
import torch
import torchvision
import torchvision.transforms as transforms 
import torch.nn as nn

from layers.Conv2D import ManualConv2d
from layers.AvgPool2d import ManualAvgPool2d
from layers.Linear import ManualLinear
from layers.Tanh import ManualTanh


class ManualLeNet5:

    def __init__(self):

        # Primera etapa convolucional
        self.conv1 = ManualConv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True
        )

        self.tanh1 = ManualTanh()

        self.pool1 = ManualAvgPool2d(
            kernel_size=2,
            stride=2
        )

        # Segunda etapa convolucional
        self.conv2 = ManualConv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=True
        )

        self.tanh2 = ManualTanh()

        self.pool2 = ManualAvgPool2d(
            kernel_size=2,
            stride=2
        )

        # Capas densas
        self.fc1 = ManualLinear(
            in_features=16 * 5 * 5,
            out_features=120,
            bias=True
        )

        self.tanh3 = ManualTanh()

        self.fc2 = ManualLinear(
            in_features=120,
            out_features=84,
            bias=True
        )

        self.tanh4 = ManualTanh()

        self.fc3 = ManualLinear(
            in_features=84,
            out_features=10,
            bias=True
        )


    def _flatten(self, x):

        flattened = []

        for batch_item in x:

            flattened_item = []

            for channel in batch_item:
                for row in channel:
                    for value in row:
                        flattened_item.append(value)

            flattened.append(flattened_item)

        return flattened


    def forward(self, x):

        x = self.conv1(x)
        x = self.tanh1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.pool2(x)

        x = self._flatten(x)

        x = self.fc1(x)
        x = self.tanh3(x)

        x = self.fc2(x)
        x = self.tanh4(x)

        x = self.fc3(x)

        return x

    def __call__(self, x):
        return self.forward(x)

    def load_pytorch_weights(self, weights_path):

        state_dict = torch.load(
            weights_path,
            map_location="cpu"
        )

        print("Keys found in state_dict:")

        for key in state_dict.keys():
            print(key)



        # Conv1
        self.conv1.weight = (
            state_dict["features.0.weight"]
            .detach()
            .cpu()
            .tolist()
        )

        self.conv1.bias = (
            state_dict["features.0.bias"]
            .detach()
            .cpu()
            .tolist()
        )

        # Conv2
        self.conv2.weight = (
            state_dict["features.3.weight"]
            .detach()
            .cpu()
            .tolist()
        )

        self.conv2.bias = (
            state_dict["features.3.bias"]
            .detach()
            .cpu()
            .tolist()
        )

        # FC1
        self.fc1.weight = (
            state_dict["classifier.1.weight"]
            .detach()
            .cpu()
            .tolist()
        )

        self.fc1.bias = (
            state_dict["classifier.1.bias"]
            .detach()
            .cpu()
            .tolist()
        )

        # FC2
        self.fc2.weight = (
            state_dict["classifier.3.weight"]
            .detach()
            .cpu()
            .tolist()
        )

        self.fc2.bias = (
            state_dict["classifier.3.bias"]
            .detach()
            .cpu()
            .tolist()
        )

        # FC3
        self.fc3.weight = (
            state_dict["classifier.5.weight"]
            .detach()
            .cpu()
            .tolist()
        )

        self.fc3.bias = (
            state_dict["classifier.5.bias"]
            .detach()
            .cpu()
            .tolist()
        )

        print("PyTorch weights loaded successfully.")

"""
if __name__ == "__main__":

    model = ManualLeNet5()

    model.load_pytorch_weights(
        "lenet5_reference.pth"
    )

    x = [
        [
            [
                [0.0 for _ in range(28)]
                for _ in range(28)
            ]
        ]
    ]

    output = model(x)

    print("Batch size:", len(output))
    print("Output size:", len(output[0]))
    print("Output:", output[0])

print("Conv1:", len(model.conv1.weight),
    len(model.conv1.weight[0]),
    len(model.conv1.weight[0][0]),
    len(model.conv1.weight[0][0][0]))

print("FC1:",
    len(model.fc1.weight),
    len(model.fc1.weight[0]))

"""
"""

if __name__ == "__main__":

    # ---------------------------------
    # 1. Cargar LeNet manual
    # ---------------------------------

    manual_model = ManualLeNet5()

    manual_model.load_pytorch_weights(
        "lenet5_reference.pth"
    )


    # ---------------------------------
    # 2. Cargar una imagen real de MNIST
    # ---------------------------------

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,)
        )
    ])

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        transform=transform,
        download=True
    )

    image, label = test_dataset[0]

    # Agregar dimensión de batch
    image = image.unsqueeze(0)

    print("Input shape:", image.shape)
    print("Real label:", label)


    # ---------------------------------
    # 3. Inferencia MANUAL
    # ---------------------------------

    manual_input = image.tolist()

    manual_output = manual_model(manual_input)

    manual_logits = manual_output[0]

    manual_prediction = max(
        range(len(manual_logits)),
        key=lambda i: manual_logits[i]
    )


    # ---------------------------------
    # 4. Mostrar resultado manual
    # ---------------------------------

    print("\nManual logits:")

    for i, value in enumerate(manual_logits):
        print(f"{i}: {value:.8f}")

    print("\nManual prediction:", manual_prediction)


    # ---------------------------------
    # 5. Crear la LeNet de referencia en PyTorch
    # ---------------------------------

    class LeNet5PyTorch(nn.Module):
        def __init__(self):
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=6,
                    kernel_size=5,
                    stride=1,
                    padding=2
                ),
                nn.Tanh(),
                nn.AvgPool2d(
                    kernel_size=2,
                    stride=2
                ),

                nn.Conv2d(
                    in_channels=6,
                    out_channels=16,
                    kernel_size=5,
                    stride=1,
                    padding=0
                ),
                nn.Tanh(),
                nn.AvgPool2d(
                    kernel_size=2,
                    stride=2
                )
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),

                nn.Linear(
                    in_features=16 * 5 * 5,
                    out_features=120
                ),
                nn.Tanh(),

                nn.Linear(
                    in_features=120,
                    out_features=84
                ),
                nn.Tanh(),

                nn.Linear(
                    in_features=84,
                    out_features=10
                )
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)

            return x

# ---------------------------------
# 6. Cargar los mismos pesos en PyTorch
# ---------------------------------

pytorch_model = LeNet5PyTorch()

state_dict = torch.load(
    "lenet5_reference.pth",
    map_location="cpu"
)

pytorch_model.load_state_dict(state_dict)

pytorch_model.eval()

# ---------------------------------
# 7. Inferencia PyTorch
# ---------------------------------

with torch.no_grad():
    pytorch_output = pytorch_model(image)

pytorch_logits = pytorch_output[0].tolist()

pytorch_prediction = pytorch_output.argmax(dim=1).item()

# ---------------------------------
# 8. Comparación PyTorch vs Manual
# ---------------------------------

print("\nComparison:")

squared_error_sum = 0.0
max_error = 0.0

for i in range(10):

    pytorch_value = pytorch_logits[i]
    manual_value = manual_logits[i]

    error = abs(
        pytorch_value - manual_value
    )

    squared_error_sum += error ** 2

    if error > max_error:
        max_error = error

    print(
        f"Class {i}: "
        f"PyTorch={pytorch_value:.8f} | "
        f"Manual={manual_value:.8f} | "
        f"Error={error:.8e}"
    )


rmse = math.sqrt(
    squared_error_sum / len(pytorch_logits)
)


print("\nReal label:", label)
print("PyTorch prediction:", pytorch_prediction)
print("Manual prediction:", manual_prediction)

print(f"\nRMSE: {rmse:.8e}")
print(f"Max absolute error: {max_error:.8e}")
"""