"""
Evaluación de LeNet-5 con DAG/XRT frente a PyTorch.

Evalúa las 10 000 imágenes del conjunto de prueba MNIST.

Métricas:
    - Accuracy PyTorch
    - Accuracy FPGA
    - RMSE global
    - Error máximo absoluto
    - Diferencias entre predicciones
    - Imágenes con error superior a la tolerancia

Ejecutar desde la raíz del proyecto:

PYTHONPATH=.:host:host/bindings python3 tests/test_lenet5_fpga.py
"""

import argparse
import math

import torch
from torch import nn
from torchvision import datasets, transforms

from fpga_graph import FpgaGraph
from models.FpgaLeNet5 import FpgaLeNet5


# ============================================================
# MODELO DE REFERENCIA PYTORCH
# ============================================================

class ReferenceLeNet5(nn.Module):

    def __init__(self):
        super().__init__()

        # Extracción de características
        self.features = nn.Sequential(
            nn.Conv2d(
                1, 6, 5,
                stride=1,
                padding=2,
                bias=True
            ),
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(
                6, 16, 5,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2),
        )

        # Clasificación
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),

            nn.Linear(120, 84),
            nn.Tanh(),

            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main():

    # --------------------------------------------------------
    # ARGUMENTOS
    # --------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        default="weights/lenet5_reference.pth"
    )

    parser.add_argument(
        "--data",
        default="./data"
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Descargar MNIST si no está disponible"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Cantidad de imágenes que se evaluarán"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # CARGAR MNIST
    # --------------------------------------------------------

    print("\nCargando conjunto de prueba MNIST...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,)
        )
    ])

    dataset = datasets.MNIST(
        root=args.data,
        train=False,
        transform=transform,
        download=args.download
    )

    total_images = min(
        args.limit,
        len(dataset)
    )

    if total_images <= 0:
        raise ValueError(
            "La cantidad de imágenes debe ser mayor que cero"
        )

    print(f"Imágenes a evaluar: {total_images}")

    # --------------------------------------------------------
    # CARGAR MODELO PYTORCH
    # --------------------------------------------------------

    print("\nCargando modelo de referencia PyTorch...")

    reference = ReferenceLeNet5().eval()

    state = torch.load(
        args.weights,
        map_location="cpu",
        weights_only=True
    )

    reference.load_state_dict(
        state,
        strict=True
    )

    # --------------------------------------------------------
    # CARGAR MODELO FPGA
    # --------------------------------------------------------

    print("Cargando modelo FPGA...")

    fpga_model = FpgaLeNet5().eval()

    fpga_model.load_pytorch_weights(
        args.weights
    )

    # --------------------------------------------------------
    # CONSTRUIR EL DAG UNA SOLA VEZ
    # --------------------------------------------------------

    print("\nConstruyendo grafo de ejecución diferida...")

    example_input = torch.zeros(
        1, 1, 28, 28,
        dtype=torch.float32
    )

    graph = FpgaGraph().build(
        fpga_model,
        example_input
    )

    print("Grafo construido correctamente.")

    # --------------------------------------------------------
    # INICIALIZAR MÉTRICAS
    # --------------------------------------------------------

    correct_pytorch = 0
    correct_fpga = 0

    different_predictions = 0
    images_above_tolerance = 0

    squared_error_sum = 0.0
    total_output_elements = 0

    global_max_error = 0.0

    # --------------------------------------------------------
    # EVALUAR LAS IMÁGENES
    # --------------------------------------------------------

    print("\nIniciando evaluación...\n")

    with torch.inference_mode():

        for index in range(total_images):

            # Obtener imagen y etiqueta
            image, real_label = dataset[index]

            # Incorporar dimensión batch
            input_tensor = image.unsqueeze(0).contiguous()

            # Ejecutar PyTorch
            expected = reference(
                input_tensor
            )

            # Ejecutar DAG en FPGA
            actual = graph.run(
                input_tensor
            )

            # -----------------------------------------------
            # VALIDACIÓN NUMÉRICA
            # -----------------------------------------------

            if actual.shape != expected.shape:
                raise RuntimeError(
                    "Forma de salida diferente en imagen "
                    f"{index}: FPGA={tuple(actual.shape)}, "
                    f"PyTorch={tuple(expected.shape)}"
                )

            difference = (
                actual - expected
            ).to(torch.float64)

            squared_error_sum += float(
                difference.square().sum()
            )

            total_output_elements += (
                difference.numel()
            )

            image_max_error = float(
                difference.abs().max()
            )

            global_max_error = max(
                global_max_error,
                image_max_error
            )

            if image_max_error > args.tolerance:
                images_above_tolerance += 1

            # -----------------------------------------------
            # PREDICCIONES
            # -----------------------------------------------

            prediction_pytorch = int(
                expected.argmax(dim=1).item()
            )

            prediction_fpga = int(
                actual.argmax(dim=1).item()
            )

            if prediction_pytorch == real_label:
                correct_pytorch += 1

            if prediction_fpga == real_label:
                correct_fpga += 1

            if prediction_pytorch != prediction_fpga:
                different_predictions += 1

            # -----------------------------------------------
            # MOSTRAR PROGRESO
            # -----------------------------------------------

            if (index + 1) % 200 == 0:

                print(
                    f"Procesadas: "
                    f"{index + 1}/{total_images}"
                )

    # --------------------------------------------------------
    # CALCULAR MÉTRICAS FINALES
    # --------------------------------------------------------

    accuracy_pytorch = (
        100.0 * correct_pytorch / total_images
    )

    accuracy_fpga = (
        100.0 * correct_fpga / total_images
    )

    global_rmse = math.sqrt(
        squared_error_sum / total_output_elements
    )

    numeric_pass = (
        images_above_tolerance == 0
    )

    # --------------------------------------------------------
    # PRESENTAR RESULTADOS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RESULTADOS DE EVALUACIÓN LENET-5 VS PYTORCH")
    print("=" * 60)

    print(
        f"Imágenes evaluadas:       {total_images}"
    )

    print(
        f"Correctas PyTorch:        {correct_pytorch}"
    )

    print(
        f"Correctas FPGA:       {correct_fpga}"
    )

    print(
        f"Accuracy PyTorch:         {accuracy_pytorch:.2f}%"
    )

    print(
        f"Accuracy FPGA:        {accuracy_fpga:.2f}%"
    )

    print(
        f"Predicciones diferentes:  {different_predictions}"
    )

    print(
        f"RMSE global:              {global_rmse:.10e}"
    )

    print(
        f"Error máximo global:      {global_max_error:.10e}"
    )

    print(
        f"Imágenes fuera tolerancia:{images_above_tolerance}"
    )

    print(
        f"Tolerancia utilizada:     {args.tolerance:.1e}"
    )

    print(
        "\nValidación numérica: "
        f"{'PASS' if numeric_pass else 'FAIL'}"
    )

    print("=" * 60)

    if not numeric_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()