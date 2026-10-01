import sys

import torch
from PIL import Image
from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights,
)

from models.FpgaMobileNetV2 import FpgaMobileNetV2


TOLERANCE = 1e-3


def main():

    if len(sys.argv) != 2:
        print(
            "Uso: python3 tests/test_mobilenetv2_fpga.py "
            "<imagen>"
        )
        raise SystemExit(1)

    image_path = sys.argv[1]

    print("======================================")
    print("MobileNetV2 - PyTorch vs FPGA")
    print("======================================")
    print("Imagen:", image_path)

    # ========================================================
    # Modelo oficial de torchvision
    # ========================================================

    weights = MobileNet_V2_Weights.IMAGENET1K_V1

    pytorch_model = mobilenet_v2(
        weights=weights
    )

    pytorch_model.eval()

    # ========================================================
    # Modelo FPGA
    # ========================================================

    fpga_model = FpgaMobileNetV2(
        num_classes=1000
    )

    fpga_model.load_pytorch_weights(
        pytorch_model
    )

    fpga_model.eval()

    # ========================================================
    # Imagen
    # ========================================================

    image = Image.open(
        image_path
    ).convert("RGB")

    preprocess = weights.transforms()

    input_tensor = preprocess(
        image
    ).unsqueeze(0)

    print(
        "Entrada:",
        tuple(input_tensor.shape)
    )

    # ========================================================
    # PyTorch
    # ========================================================

    print()
    print("Ejecutando MobileNetV2 PyTorch...")

    with torch.inference_mode():

        output_torch = pytorch_model(
            input_tensor
        )

    # ========================================================
    # FPGA
    # ========================================================

    print()
    print("Ejecutando MobileNetV2 FPGA...")

    with torch.inference_mode():

        output_fpga = fpga_model(
            input_tensor,
            verbose=True
        )

    # ========================================================
    # Comparación numérica
    # ========================================================

    difference = (
        output_fpga
        - output_torch
    )

    rmse = torch.sqrt(
        torch.mean(
            difference ** 2
        )
    ).item()

    max_error = torch.max(
        torch.abs(difference)
    ).item()

    print()
    print("======================================")
    print("Comparación de logits")
    print("======================================")

    print(
        "Salida PyTorch:",
        tuple(output_torch.shape)
    )

    print(
        "Salida FPGA:   ",
        tuple(output_fpga.shape)
    )

    print(
        f"RMSE:          {rmse:.10e}"
    )

    print(
        f"Error máximo:  {max_error:.10e}"
    )

    # ========================================================
    # Top-5
    # ========================================================

    categories = weights.meta[
        "categories"
    ]

    probabilities_torch = torch.softmax(
        output_torch[0],
        dim=0
    )

    probabilities_fpga = torch.softmax(
        output_fpga[0],
        dim=0
    )

    top5_torch = torch.topk(
        probabilities_torch,
        5
    )

    top5_fpga = torch.topk(
        probabilities_fpga,
        5
    )

    print()
    print("======================================")
    print("Top-5 PyTorch")
    print("======================================")

    for probability, index in zip(
        top5_torch.values,
        top5_torch.indices
    ):

        print(
            f"{categories[index.item()]:30s} "
            f"{probability.item() * 100:8.4f}%"
        )

    print()
    print("======================================")
    print("Top-5 FPGA")
    print("======================================")

    for probability, index in zip(
        top5_fpga.values,
        top5_fpga.indices
    ):

        print(
            f"{categories[index.item()]:30s} "
            f"{probability.item() * 100:8.4f}%"
        )

    # ========================================================
    # Validación
    # ========================================================

    top1_torch = (
        top5_torch.indices[0].item()
    )

    top1_fpga = (
        top5_fpga.indices[0].item()
    )

    same_prediction = (
        top1_torch == top1_fpga
    )

    numerical_pass = (
        rmse <= TOLERANCE
    )

    passed = (
        numerical_pass
        and same_prediction
    )

    print()
    print("======================================")
    print("Resultado")
    print("======================================")

    print(
        "Top-1 PyTorch:",
        categories[top1_torch]
    )

    print(
        "Top-1 FPGA:   ",
        categories[top1_fpga]
    )

    print(
        "Predicción igual:",
        same_prediction
    )

    print(
        "RMSE <= 1e-3:",
        numerical_pass
    )

    print()
    print(
        "RESULTADO GENERAL:",
        "PASS" if passed else "FAIL"
    )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
