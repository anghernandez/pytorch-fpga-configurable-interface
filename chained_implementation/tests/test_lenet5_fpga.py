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

import time
import fpga_dag

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


# FUNCIÓN PRINCIPAL

def main():
    # --------------------------------------------------------
    # Configuración de la prueba
    # --------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        default="weights/lenet5_reference.pth",
    )
    parser.add_argument("--data", default="./data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--tolerance", type=float, default=1e-4)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--inflight-batches",
        type=int,
        default=4,
        help="Batches entregados juntos al planificador",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rest", type=float, default=5.0)
    parser.add_argument(
        "--mode",
        choices=("sequential", "pipeline", "both"),
        default="both",
    )

    args = parser.parse_args()

    if args.limit <= 0:
        parser.error("--limit debe ser positivo")
    if args.batch_size <= 0 or args.inflight_batches <= 0:
        parser.error("Los tamaños de batch y ventana deben ser positivos")
    if args.repeats <= 0 or args.warmup < 0:
        parser.error("--repeats debe ser positivo y --warmup no negativo")
    if not math.isfinite(args.rest) or args.rest < 0:
        parser.error("--rest debe ser finito y no negativo")
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance debe ser finita y positiva")

    # --------------------------------------------------------
    # Preparar imágenes fuera del cronómetro
    # --------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,),
        ),
    ])

    dataset = datasets.MNIST(
        root=args.data,
        train=False,
        transform=transform,
        download=args.download,
    )

    total_images = min(args.limit, len(dataset))

    if total_images == 0:
        raise ValueError("El conjunto de evaluación está vacío")

    samples = [dataset[index] for index in range(total_images)]
    images = torch.stack([sample[0] for sample in samples])
    labels = torch.tensor([sample[1] for sample in samples])

    # Todos los batches deben coincidir con la forma del grafo.
    # Completar únicamente el último con ceros si fuera necesario.
    # Esas muestras adicionales no participan en las métricas.
    input_batches = []
    valid_counts = []

    for start in range(0, total_images, args.batch_size):
        part = images[start:start + args.batch_size]
        valid = part.shape[0]

        batch = torch.zeros(
            args.batch_size, 1, 28, 28,
            dtype=torch.float32,
        )
        batch[:valid].copy_(part)

        input_batches.append(batch)
        valid_counts.append(valid)

    # --------------------------------------------------------
    # Cargar modelo y calcular la referencia PyTorch
    # --------------------------------------------------------
    reference = ReferenceLeNet5().eval()

    state = torch.load(
        args.weights,
        map_location="cpu",
        weights_only=True,
    )
    reference.load_state_dict(state, strict=True)

    with torch.inference_mode():
        expected = torch.cat([
            reference(batch)[:valid]
            for batch, valid in zip(input_batches, valid_counts)
        ])

    prediction_pytorch = expected.argmax(dim=1)
    accuracy_pytorch = (
        100.0 *
        prediction_pytorch.eq(labels).sum().item() /
        total_images
    )

    fpga_model = FpgaLeNet5().eval()
    fpga_model.load_pytorch_weights(args.weights)

    modes = (
        ["sequential", "pipeline"]
        if args.mode == "both"
        else [args.mode]
    )

    timings = {mode: [] for mode in modes}

    print("\n" + "-" * 60)
    print("LeNet-5: ejecución secuencial y pipeline")
    print(f"Imágenes reales: {total_images}")
    print(f"Muestras por batch: {args.batch_size}")
    print(f"Batches totales: {len(input_batches)}")
    print(f"Batches por ventana: {args.inflight_batches}")
    print(f"Repeticiones medidas por modo: {args.repeats}")
    print(f"Accuracy PyTorch: {accuracy_pytorch:.2f}%")

    padded = len(input_batches) * args.batch_size - total_images
    print(f"Muestras de relleno: {padded}")
    print("Tiempo: carga + ejecución + descarga y conversión")
    print("-" * 60)

    try:
        # ----------------------------------------------------
        # Construir una sola vez para el batch configurado
        # ----------------------------------------------------
        with torch.inference_mode():
            graph = FpgaGraph().build(
                fpga_model,
                torch.zeros(
                    args.batch_size, 1, 28, 28,
                    dtype=torch.float32,
                ),
            )

        # Después de cada ejecución deben permanecer únicamente
        # los buffers que ya existían tras construir el grafo.
        persistent_count = fpga_dag.buffer_count()

        def evaluate(mode):
            elapsed = 0.0
            collected = []

            with torch.inference_mode():
                # Procesar ventanas para no cargar todas las
                # entradas del conjunto simultáneamente en XRT.
                for start in range(
                    0,
                    len(input_batches),
                    args.inflight_batches,
                ):
                    window = input_batches[
                        start:start + args.inflight_batches
                    ]

                    begin = time.perf_counter()
                    outputs = graph.run_batches(window, mode=mode) #desde python acá se elige el pipeline, desde acá se envian los batches 
                    elapsed += time.perf_counter() - begin

                    if len(outputs) != len(window):
                        raise RuntimeError(
                            "Cantidad incorrecta de salidas"
                        )

                    # Validar y retirar el relleno fuera del tiempo.
                    for offset, output in enumerate(outputs):
                        if tuple(output.shape) != (
                            args.batch_size, 10
                        ):
                            raise RuntimeError(
                                "Forma de salida incorrecta: "
                                f"{tuple(output.shape)}"
                            )

                        valid = valid_counts[start + offset]
                        collected.append(output[:valid])

                    if fpga_dag.buffer_count() != persistent_count:
                        raise RuntimeError(
                            "Quedaron activaciones activas "
                            "después de descargar las salidas"
                        )

                actual = torch.cat(collected)

                if actual.shape != expected.shape:
                    raise RuntimeError(
                        "La forma final no coincide con PyTorch"
                    )

                if not torch.isfinite(actual).all().item():
                    raise RuntimeError(
                        "La salida contiene valores no finitos"
                    )

                difference = (
                    actual.to(torch.float64) -
                    expected.to(torch.float64)
                )

                per_image_error = difference.abs().amax(dim=1)
                maximum = per_image_error.max().item()
                outside = (
                    per_image_error > args.tolerance
                ).sum().item()

                rmse = difference.square().mean().sqrt().item()

                predictions = actual.argmax(dim=1)
                accuracy = (
                    100.0 *
                    predictions.eq(labels).sum().item() /
                    total_images
                )
                different = predictions.ne(
                    prediction_pytorch
                ).sum().item()

            if outside:
                raise RuntimeError(
                    f"{mode}: {outside} imágenes fuera de "
                    f"tolerancia; error máximo={maximum:.6e}"
                )

            return elapsed, accuracy, rmse, maximum, different

        # ----------------------------------------------------
        # Calentar y medir alternando el orden de los modos
        # ----------------------------------------------------
        rounds = args.warmup + args.repeats

        for round_index in range(rounds):
            warmup = round_index < args.warmup

            ordered_modes = (
                modes if round_index % 2 == 0 else modes[::-1]
            )

            for mode in ordered_modes:
                if args.rest:
                    time.sleep(args.rest)

                elapsed, accuracy, rmse, maximum, different = (
                    evaluate(mode)
                )

                if warmup:
                    label = (
                        f"Calentamiento {round_index + 1}"
                    )
                else:
                    number = round_index - args.warmup + 1
                    label = f"Repetición {number}"
                    timings[mode].append(elapsed)

                print(
                    f"{label} | {mode} | "
                    f"{elapsed * 1000:.3f} ms | "
                    f"Accuracy: {accuracy:.2f}% | "
                    f"RMSE: {rmse:.3e} | "
                    f"Máximo: {maximum:.3e} | "
                    f"Predicciones diferentes: {different}",
                    flush=True,
                )

        # ----------------------------------------------------
        # Resumen de mediciones validadas
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
        # Liberar XRT explícitamente antes de salir del proceso.
        fpga_dag.shutdown()

    print("PASS: validación numérica y limpieza completadas.")

if __name__ == "__main__":
    main()