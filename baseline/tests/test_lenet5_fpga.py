# El objetivo de este test es comparar LeNet-5 en PyTorch
# con LeNet-5 que utiliza los kernels de FPGA.

import argparse
from time import perf_counter

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import fpga_kernels

from torch.utils.data import DataLoader, Subset

from models.LeNet5_reference import LeNet5
from models.fpgaLeNet5 import FpgaLeNet5


# ============================================================
# Configuración
# ============================================================

TOLERANCE = 1e-3
WEIGHTS_PATH = "weights/lenet5_reference.pth"


# ============================================================
# Argumentos de ejecución
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--limit",
    type=int,
    default=128,
    help="Cantidad de imágenes MNIST que se evaluarán"
)

parser.add_argument(
    "--batch-size",
    type=int,
    choices=[1, 8, 32, 64, 128],
    default=1,
    help="Cantidad de imágenes por lote"
)

args = parser.parse_args()

if args.limit <= 0:
    raise ValueError("--limit debe ser mayor que cero")

BATCH_SIZE = args.batch_size


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

# Limitar la cantidad de imágenes solicitadas.
total_images_to_evaluate = min(
    args.limit,
    len(test_dataset)
)

test_dataset = Subset(
    test_dataset,
    range(total_images_to_evaluate)
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

        fpga_kernels.profile_begin(capacity=11)

        try:
            t0 = perf_counter()

            fpga_output = fpga_model(images)

            t1 = perf_counter()

        except BaseException:
            fpga_kernels.profile_cancel()
            raise

        profile = fpga_kernels.profile_end()

        if profile["operation_count"] != 11:
            raise RuntimeError(
                "Se esperaban 11 operaciones registradas, "
                f"pero se obtuvieron {profile['operation_count']}"
            )

        print(
            f"\n[Baseline] Llamada completa: "
            f"{(t1 - t0) * 1000:.3f} ms"
        )

        print(
            "[Baseline] Primera sincronización → última recepción: "
            f"{profile['latencia_red_ms']:.3f} ms"
        )

        for index, layer in enumerate(profile["layers"], start=1):
            print(
                f"  {index:02d}. {layer['operation']}: "
                f"{layer['latencia_capa_ms']:.3f} ms"
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
    sum_squared_error / total_logit_values
)

pytorch_accuracy = (
    100.0 * pytorch_correct / total_images
)

fpga_accuracy = (
    100.0 * fpga_correct / total_images
)

prediction_agreement = (
    100.0 * same_predictions / total_images
)

different_predictions = (
    total_images - same_predictions
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

