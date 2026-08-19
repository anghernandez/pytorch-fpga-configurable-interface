import math

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from LeNet5_manual import ManualLeNet5

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
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    


# Modelo PyTorch
pytorch_model = LeNet5PyTorch()

state_dict = torch.load(
    "lenet5_reference.pth",
    map_location="cpu"
)

pytorch_model.load_state_dict(state_dict)
pytorch_model.eval()


# Modelo manual
manual_model = ManualLeNet5()

manual_model.load_pytorch_weights(
    "lenet5_reference.pth"
)


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

#num_images = 1000
num_images = len(test_dataset)

print("Test dataset size:", len(test_dataset))
print("Images to compare:", num_images)

image, label = test_dataset[0]

print("First image shape:", image.shape)
print("First image label:", label)


# Métricas
same_predictions = 0

pytorch_correct = 0
manual_correct = 0

rmse_sum = 0.0
max_rmse = 0.0
max_absolute_error = 0.0


for index in range(num_images):

    image, label = test_dataset[index]

    # Agregar dimensión de batch:
    # [1, 28, 28] -> [1, 1, 28, 28]
    image = image.unsqueeze(0)

    # -----------------------------
    # Inferencia PyTorch
    # -----------------------------

    with torch.no_grad():
        pytorch_output = pytorch_model(image)

    pytorch_logits = pytorch_output[0].tolist()

    pytorch_prediction = (
        pytorch_output.argmax(dim=1).item()
    )


    # -----------------------------
    # Inferencia manual
    # -----------------------------

    manual_input = image.tolist()

    manual_output = manual_model(manual_input)

    manual_logits = manual_output[0]

    manual_prediction = max(
        range(len(manual_logits)),
        key=lambda i: manual_logits[i]
    )


    # -----------------------------
    # Comparar predicciones
    # -----------------------------

    if pytorch_prediction == manual_prediction:
        same_predictions += 1

    if pytorch_prediction == label:
        pytorch_correct += 1

    if manual_prediction == label:
        manual_correct += 1


    # -----------------------------
    # Comparar logits
    # -----------------------------

    squared_error_sum = 0.0
    image_max_error = 0.0

    for pytorch_value, manual_value in zip(
        pytorch_logits,
        manual_logits
    ):

        error = abs(
            pytorch_value - manual_value
        )

        squared_error_sum += error ** 2

        if error > image_max_error:
            image_max_error = error


    image_rmse = math.sqrt(
        squared_error_sum / len(pytorch_logits)
    )


    # Acumular métricas
    rmse_sum += image_rmse

    if image_rmse > max_rmse:
        max_rmse = image_rmse

    if image_max_error > max_absolute_error:
        max_absolute_error = image_max_error


    # Mostrar progreso
    print(
        f"Image {index + 1:3}/{num_images} | "
        f"Label: {label} | "
        f"PyTorch: {pytorch_prediction} | "
        f"Manual: {manual_prediction} | "
        f"RMSE: {image_rmse:.3e}"
    )

    average_rmse = rmse_sum / num_images

pytorch_accuracy = (
    100 * pytorch_correct / num_images
)

manual_accuracy = (
    100 * manual_correct / num_images
)


print("\n" + "=" * 55)
print("LeNet-5 PyTorch vs Manual - Test Summary")
print("=" * 55)

print(f"Images tested:             {num_images}")

print(
    f"Same predictions:          "
    f"{same_predictions}/{num_images}"
)

print(
    f"PyTorch accuracy:          "
    f"{pytorch_accuracy:.2f}%"
)

print(
    f"Manual accuracy:           "
    f"{manual_accuracy:.2f}%"
)

print(
    f"Average RMSE:              "
    f"{average_rmse:.8e}"
)

print(
    f"Maximum RMSE:              "
    f"{max_rmse:.8e}"
)

print(
    f"Maximum absolute error:    "
    f"{max_absolute_error:.8e}"
)