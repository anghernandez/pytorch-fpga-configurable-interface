import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader


from models.LeNet5_reference import LeNet5
from models.fpgaLeNet5 import FpgaLeNet5


# ============================================================
# Configuración
# ============================================================

BATCH_SIZE = 64
TOLERANCE = 1e-3

WEIGHTS_PATH = "weights/lenet5_reference.pth"


# ============================================================
# Dataset MNIST
# ============================================================

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.1307,),
        std=(0.3081,)
    )
])


test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=mnist_transform,
    download=True
)


test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Modelo PyTorch de referencia
# ============================================================

pytorch_model = LeNet5(
    num_classes=10
)

state_dict = torch.load(
    WEIGHTS_PATH,
    map_location="cpu",
    weights_only=True
)

pytorch_model.load_state_dict(
    state_dict
)

pytorch_model.eval()


# ============================================================
# Modelo FPGA
# ============================================================

fpga_model = FpgaLeNet5()

fpga_model.load_pytorch_weights(
    WEIGHTS_PATH
)

fpga_model.eval()


# ============================================================
# Evaluación
# ============================================================

total_images = 0

pytorch_correct = 0
fpga_correct = 0

same_predictions = 0

sum_squared_error = 0.0
total_logit_values = 0

max_error = 0.0


print()
print("========================================")
print("Evaluación LeNet-5: PyTorch vs FPGA")
print("========================================")
print(f"Batch size: {BATCH_SIZE}")
print(f"Imágenes MNIST Test: {len(test_dataset)}")
print()


with torch.no_grad():

    for batch_index, (images, labels) in enumerate(
        test_loader,
        start=1
    ):

        # ----------------------------------------------------
        # PyTorch
        # ----------------------------------------------------

        pytorch_output = pytorch_model(
            images
        )


        # ----------------------------------------------------
        # FPGA
        # ----------------------------------------------------

        fpga_output = fpga_model(
            images
        )


        # ----------------------------------------------------
        # Predicciones
        # ----------------------------------------------------

        pytorch_predictions = (
            pytorch_output.argmax(dim=1)
        )

        fpga_predictions = (
            fpga_output.argmax(dim=1)
        )


        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        pytorch_correct += (
            pytorch_predictions == labels
        ).sum().item()

        fpga_correct += (
            fpga_predictions == labels
        ).sum().item()


        # ----------------------------------------------------
        # Coincidencia CPU / FPGA
        # ----------------------------------------------------

        same_predictions += (
            pytorch_predictions == fpga_predictions
        ).sum().item()


        # ----------------------------------------------------
        # Error de logits
        # ----------------------------------------------------

        difference = (
            fpga_output - pytorch_output
        )

        sum_squared_error += (
            difference.double()
            .pow(2)
            .sum()
            .item()
        )

        total_logit_values += (
            difference.numel()
        )

        batch_max_error = (
            difference.abs()
            .max()
            .item()
        )

        max_error = max(
            max_error,
            batch_max_error
        )


        # ----------------------------------------------------
        # Contador
        # ----------------------------------------------------

        total_images += labels.size(0)


        print(
            f"\rBatch {batch_index:3d}/{len(test_loader)} "
            f"| Imágenes: {total_images:5d}/{len(test_dataset)}",
            end="",
            flush=True
        )


print()
print()


# ============================================================
# Resultados finales
# ============================================================

global_rmse = np.sqrt(
    sum_squared_error
    / total_logit_values
)

pytorch_accuracy = (
    100.0
    * pytorch_correct
    / total_images
)

fpga_accuracy = (
    100.0
    * fpga_correct
    / total_images
)

prediction_agreement = (
    100.0
    * same_predictions
    / total_images
)

different_predictions = (
    total_images
    - same_predictions
)


print("========================================")
print("Resultados")
print("========================================")

print(
    f"Imágenes evaluadas:          "
    f"{total_images}"
)

print()

print(
    f"Accuracy PyTorch:            "
    f"{pytorch_accuracy:.4f}%"
)

print(
    f"Accuracy FPGA:               "
    f"{fpga_accuracy:.4f}%"
)

print()

print(
    f"RMSE global de logits:       "
    f"{global_rmse:.10e}"
)

print(
    f"Error máximo de logits:      "
    f"{max_error:.10e}"
)

print()

print(
    f"Predicciones iguales:        "
    f"{same_predictions}/{total_images}"
)

print(
    f"Predicciones diferentes:     "
    f"{different_predictions}/{total_images}"
)

print(
    f"Coincidencia de predicción:  "
    f"{prediction_agreement:.4f}%"
)

print()

"""
# ============================================================
# Criterio numérico del proyecto
# ============================================================

if global_rmse <= TOLERANCE:

    print(
        f"RMSE <= {TOLERANCE}: PASS"
    )

else:

    print(
        f"RMSE <= {TOLERANCE}: FAIL"
    )
"""