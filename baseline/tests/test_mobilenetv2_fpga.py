import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from models.FpgaMobileNetV2 import FpgaMobileNetV2
import time
from collections import Counter
import fpga_kernels

TOLERANCE = 1e-3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=[1, 8, 16,  32, 64],
        default=1,
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("test_images/imagenet128"),
    )
    args = parser.parse_args()

    # Selección de imágenes distintas para un único batch.
    with (args.image_dir / "labels.csv").open() as file:
        rows = list(csv.DictReader(file))[:args.batch_size]

    if len(rows) != args.batch_size:
        raise ValueError("No hay suficientes imágenes para este batch")

    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    images = []
    for row in rows:
        with Image.open(args.image_dir / row["image"]) as image:
            images.append(preprocess(image.convert("RGB")))

    input_tensor = torch.stack(images, dim=0)
    labels = torch.tensor([int(row["label"]) for row in rows])

    print("Entrada:", tuple(input_tensor.shape))

    pytorch_model = mobilenet_v2(weights=weights).eval()

    fpga_model = FpgaMobileNetV2(num_classes=1000)
    fpga_model.load_pytorch_weights(pytorch_model)
    fpga_model.eval()

    with torch.inference_mode():
        print("\nEjecutando PyTorch...")
        output_torch = pytorch_model(input_tensor)

        print("\nCalentamiento FPGA...")
        warmup_output = fpga_model(input_tensor, verbose=False)
        del warmup_output

        print("\nEjecutando FPGA con perfil...")
        fpga_kernels.profile_begin(256)

        try:
            start = time.perf_counter()

            output_fpga = fpga_model(
                input_tensor,
                verbose=False,
            )

            model_ms = (time.perf_counter() - start) * 1000
            profile = fpga_kernels.profile_end()

        except BaseException:
            fpga_kernels.profile_cancel()
            raise

        print(f"Llamada completa desde Python: {model_ms:.3f} ms")
        print("Latencia de red XRT (ms):", profile["latencia_red_ms"])
        print("Operaciones registradas:", profile["operation_count"])

        counts = Counter(
            layer["operation"] for layer in profile["layers"]
        )

        for operation, count in sorted(counts.items()):
            print(f"  {operation}: {count}")

    expected_shape = (args.batch_size, 1000)
    if (
        tuple(output_torch.shape) != expected_shape
        or tuple(output_fpga.shape) != expected_shape
    ):
        raise RuntimeError(
            f"Se esperaba {expected_shape}; "
            f"PyTorch={tuple(output_torch.shape)}, "
            f"FPGA={tuple(output_fpga.shape)}"
        )

    if not (
        torch.isfinite(output_torch).all().item()
        and torch.isfinite(output_fpga).all().item()
    ):
        raise RuntimeError("Hay valores NaN o infinitos en las salidas")

    # Comparación sobre todas las imágenes y todos los logits.
    difference = output_fpga - output_torch
    rmse = difference.square().mean().sqrt().item()
    max_error = difference.abs().max().item()

    # RMSE individual para detectar errores en una imagen del batch.
    rmse_per_image = difference.square().mean(dim=1).sqrt()

    predictions_torch = output_torch.argmax(dim=1)
    predictions_fpga = output_fpga.argmax(dim=1)

    matches = (predictions_torch == predictions_fpga).sum().item()
    correct_torch = (predictions_torch == labels).sum().item()
    correct_fpga = (predictions_fpga == labels).sum().item()

    print("\nPredicciones:")
    for i, row in enumerate(rows):
        real = categories[labels[i].item()]
        pred_torch = categories[predictions_torch[i].item()]
        pred_fpga = categories[predictions_fpga[i].item()]
        print(
            f'{row["image"]} | Real: {real} | '
            f"PyTorch: {pred_torch} | FPGA: {pred_fpga}"
        )

    print(f"\nRMSE global:           {rmse:.10e}")
    print(f"Mayor RMSE individual: {rmse_per_image.max().item():.10e}")
    print(f"Error máximo:          {max_error:.10e}")
    print(f"Predicciones iguales:  {matches}/{args.batch_size}")
    print(f"Accuracy PyTorch:      {100 * correct_torch / args.batch_size:.2f}%")
    print(f"Accuracy FPGA:         {100 * correct_fpga / args.batch_size:.2f}%")

    passed = (
        (rmse_per_image <= TOLERANCE).all().item()
        and matches == args.batch_size
    )

    print("\nRESULTADO GENERAL:", "PASS" if passed else "FAIL")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()