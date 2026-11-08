"""MobileNetV2: comparación secuencial/pipeline y validación con PyTorch.

El tiempo incluye carga de entradas, ejecución, descarga y conversión.
La preparación de imágenes y la validación quedan fuera del cronómetro.
"""

import argparse
import csv
import math
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

import fpga_dag
from fpga_graph import FpgaGraph
from models.FpgaMobileNetV2 import FpgaMobileNetV2


def main():
    # --------------------------------------------------------
    # Configuración
    # --------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data", type=Path,
        default=Path("test_images/imagenet128"),
    )
    parser.add_argument(
        "--weights", type=Path, default=None,
        help="state_dict local; si se omite, usa pesos oficiales",
    )
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--inflight-batches", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rest", type=float, default=5.0)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--mode",
        choices=("sequential", "pipeline", "both"),
        default="both",
    )
    args = parser.parse_args()

    if min(
        args.limit, args.batch_size,
        args.inflight_batches, args.repeats,
    ) <= 0:
        parser.error("Imágenes, batch, ventana y repeticiones deben ser positivos")

    if args.warmup < 0:
        parser.error("--warmup no puede ser negativo")

    if not math.isfinite(args.rest) or args.rest < 0:
        parser.error("--rest debe ser finito y no negativo")

    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance debe ser finita y positiva")

    labels_path = args.data / "labels.csv"
    if not labels_path.is_file():
        parser.error(f"No existe: {labels_path}")

    # --------------------------------------------------------
    # Leer el manifiesto y preparar imágenes fuera del tiempo
    # --------------------------------------------------------
    with labels_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not {"image", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("labels.csv necesita las columnas image,label")

        rows = list(reader)[:args.limit]

    if not rows:
        raise ValueError("El conjunto está vacío")

    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    images = []
    labels = []

    for row in rows:
        image_path = args.data / row["image"]
        label = int(row["label"])

        if not 0 <= label < 1000:
            raise ValueError(f"Etiqueta fuera de rango: {label}")

        with Image.open(image_path) as image:
            images.append(preprocess(image.convert("RGB")))

        labels.append(label)

    labels = torch.tensor(labels, dtype=torch.long)
    total_images = len(images)

    # El grafo tiene una forma fija. Rellenar solo el último batch.
    # El relleno no participa en accuracy ni validación numérica.
    input_batches = []
    valid_counts = []

    for start in range(0, total_images, args.batch_size):
        part = images[start:start + args.batch_size]
        batch = torch.zeros(
            args.batch_size, *images[0].shape,
            dtype=torch.float32,
        )
        batch[:len(part)].copy_(torch.stack(part))
        input_batches.append(batch)
        valid_counts.append(len(part))

    del images

    # --------------------------------------------------------
    # Cargar pesos y calcular la referencia fuera del tiempo
    # --------------------------------------------------------
    if args.weights is None:
        reference = mobilenet_v2(weights=weights).eval()
    else:
        reference = mobilenet_v2(weights=None).eval()
        state = torch.load(
            args.weights, map_location="cpu", weights_only=True,
        )
        reference.load_state_dict(state, strict=True)

    with torch.inference_mode():
        expected = torch.cat([
            reference(batch)[:valid]
            for batch, valid in zip(input_batches, valid_counts)
        ])

    if not torch.isfinite(expected).all().item():
        raise RuntimeError("La referencia contiene valores no finitos")

    reference_predictions = expected.argmax(dim=1)
    reference_accuracy = (
        100.0 * reference_predictions.eq(labels).sum().item()
        / total_images
    )

    fpga_model = FpgaMobileNetV2(num_classes=1000).eval()
    fpga_model.load_pytorch_weights(reference)

    modes = (
        ["sequential", "pipeline"]
        if args.mode == "both" else [args.mode]
    )
    timings = {mode: [] for mode in modes}

    print("\n" + "-" * 60)
    print("MobileNetV2: ejecución secuencial y pipeline")
    print(f"Imágenes reales: {total_images}")
    print(f"Muestras por batch: {args.batch_size}")
    print(f"Batches totales: {len(input_batches)}")
    print(f"Batches por ventana: {args.inflight_batches}")
    print(
        "Muestras de relleno: "
        f"{len(input_batches) * args.batch_size - total_images}"
    )
    print(f"Accuracy PyTorch: {reference_accuracy:.2f}%")
    print("Tiempo: carga + ejecución + descarga y conversión")
    print("-" * 60, flush=True)

    try:
        # ----------------------------------------------------
        # Construir el grafo y cargar parámetros una sola vez
        # ----------------------------------------------------
        with torch.inference_mode():
            graph = FpgaGraph().build(
                fpga_model,
                torch.zeros_like(input_batches[0]),
            )

        persistent_count = fpga_dag.buffer_count()

        def evaluate(mode):
            elapsed = 0.0
            collected = []

            with torch.inference_mode():
                for start in range(
                    0, len(input_batches), args.inflight_batches,
                ):
                    window = input_batches[
                        start:start + args.inflight_batches
                    ]

                    # Medir únicamente la llamada a la interfaz.
                    begin = time.perf_counter()
                    outputs = graph.run_batches(window, mode=mode)
                    elapsed += time.perf_counter() - begin

                    if len(outputs) != len(window):
                        raise RuntimeError("Cantidad incorrecta de salidas")

                    # Validar formas y retirar relleno fuera del tiempo.
                    for offset, output in enumerate(outputs):
                        if tuple(output.shape) != (args.batch_size, 1000):
                            raise RuntimeError(
                                f"Forma incorrecta: {tuple(output.shape)}"
                            )

                        valid = valid_counts[start + offset]
                        collected.append(output[:valid])

                    if fpga_dag.buffer_count() != persistent_count:
                        raise RuntimeError(
                            "Quedaron activaciones activas tras la ventana"
                        )

                actual = torch.cat(collected)

                if actual.shape != expected.shape:
                    raise RuntimeError("La forma final difiere de PyTorch")

                if not torch.isfinite(actual).all().item():
                    raise RuntimeError("La salida contiene valores no finitos")

                difference = actual.double() - expected.double()
                per_image_error = difference.abs().amax(dim=1)
                maximum = per_image_error.max().item()
                rmse = difference.square().mean().sqrt().item()
                outside = (
                    per_image_error > args.tolerance
                ).sum().item()

                predictions = actual.argmax(dim=1)
                accuracy = (
                    100.0 * predictions.eq(labels).sum().item()
                    / total_images
                )
                different = predictions.ne(
                    reference_predictions
                ).sum().item()

            if outside:
                raise RuntimeError(
                    f"{mode}: {outside} imágenes fuera de tolerancia; "
                    f"máximo={maximum:.6e}"
                )

            return elapsed, accuracy, rmse, maximum, different

        # ----------------------------------------------------
        # Calentamiento y mediciones con orden alternado
        # ----------------------------------------------------
        for round_index in range(args.warmup + args.repeats):
            warmup = round_index < args.warmup
            ordered_modes = (
                modes if round_index % 2 == 0 else modes[::-1]
            )

            for mode in ordered_modes:
                if args.rest:
                    time.sleep(args.rest)

                elapsed, accuracy, rmse, maximum, different = evaluate(mode)

                if warmup:
                    label = f"Calentamiento {round_index + 1}"
                else:
                    number = round_index - args.warmup + 1
                    label = f"Repetición {number}"
                    timings[mode].append(elapsed)

                print(
                    f"{label} | {mode} | {elapsed * 1000:.3f} ms | "
                    f"Accuracy: {accuracy:.2f}% | RMSE: {rmse:.3e} | "
                    f"Máximo: {maximum:.3e} | "
                    f"Predicciones diferentes: {different}",
                    flush=True,
                )

        # ----------------------------------------------------
        # Resumen: solo repeticiones medidas y validadas
        # ----------------------------------------------------
        averages = {
            mode: sum(values) / len(values)
            for mode, values in timings.items()
        }

        print("\n" + "-" * 60)

        for mode, average in averages.items():
            print(
                f"{mode}: {average * 1000:.3f} ms | "
                f"{average * 1000 / total_images:.3f} ms/imagen | "
                f"{total_images / average:.2f} imágenes/s"
            )

        if args.mode == "both":
            sequential = averages["sequential"]
            pipeline = averages["pipeline"]

            print(f"Aceleración: {sequential / pipeline:.3f}x")
            print(
                "Reducción del tiempo: "
                f"{100.0 * (1.0 - pipeline / sequential):.2f}%"
            )

    finally:
        # Liberar recursos incluso si falla la construcción o ejecución.
        fpga_dag.shutdown()

    print("PASS: validación numérica y limpieza completadas.")


if __name__ == "__main__":
    main()