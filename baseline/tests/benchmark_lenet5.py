"""
Benchmark del baseline LeNet-5.

- 128 imágenes distintas de MNIST.
- Batches: 1, 8, 32, 64 y 128.
- Cinco repeticiones por configuración.
- Calentamiento excluido de las mediciones.
- Matriz de confusión promedio por batch.
- Media y desviación estándar de tiempos y métricas.
- Perfilado opcional de capas en repeticiones separadas.

Los tiempos incluyen wrappers y gestión del host.
No representan exclusivamente el tiempo del hardware FPGA.

Pendientes:
- Potencia.
- Latencia delimitada por transferencias XRT.
- Separación del overhead.
"""

import argparse
import csv
import hashlib
import json
import platform

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from models.LeNet5_reference import LeNet5
from models.fpgaLeNet5 import FpgaLeNet5


# ============================================================
# Configuración
# ============================================================

NUM_IMAGES = 128
NUM_CLASSES = 10
TOLERANCE = 1e-3


# ============================================================
# Utilidades
# ============================================================

def write_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def confusion(labels, predictions):
    return np.bincount(
        NUM_CLASSES * labels + predictions,
        minlength=NUM_CLASSES * NUM_CLASSES
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def add_statistics(destination, metric, values):
    destination[metric + "_mean"] = float(
        np.mean(values)
    )
    destination[metric + "_std"] = float(
        np.std(values, ddof=1)
    )


# ============================================================
# Ejecución
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        choices=[1, 8, 32, 64, 128],
        default=[1, 8, 32, 64, 128]
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Pasadas completas de calentamiento por batch"
    )

    parser.add_argument(
        "--weights",
        default="weights/lenet5_reference.pth"
    )

    parser.add_argument(
        "--output",
        default="results/lenet5"
    )

    parser.add_argument(
        "--profile-layers",
        action="store_true",
        help=(
            "Mide capas en pasadas adicionales. "
            "Usa la misma cantidad de repeticiones que --repeats."
        )
    )

    args = parser.parse_args()

    if args.repeats < 2:
        parser.error(
            "Se necesitan al menos dos repeticiones"
        )

    if args.warmup < 1:
        parser.error(
            "Se necesita al menos una pasada de calentamiento"
        )

    # Evita evaluar accidentalmente un batch duplicado.
    args.batch_sizes = list(dict.fromkeys(args.batch_sizes))

    # ========================================================
    # Carpeta de resultados
    # ========================================================

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )

    out = Path(args.output) / timestamp

    out.mkdir(
        parents=True,
        exist_ok=False
    )

    metadata = {
        "arguments": vars(args),
        "image_indices": list(range(NUM_IMAGES)),
        "dataset": "MNIST test",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "weights_sha256": hashlib.sha256(
            Path(args.weights).read_bytes()
        ).hexdigest(),
        "timing": (
            "Llamada Python sincrónica a fpga_model; "
            "incluye wrappers y gestión del host"
        ),
        "power_measured": False,
        "xrt_transfer_latency_measured": False,
        "confusion_matrix": (
            "Promedio por elemento entre repeticiones; "
            "filas = etiqueta real, columnas = predicción"
        ),
        "layer_timing": (
            "Repeticiones separadas de las mediciones principales; "
            "incluye wrapper y gestión del host"
        ),
        "std_ddof": 1,
    }

    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # ========================================================
    # Dataset
    # ========================================================

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,),
            (0.3081,)
        )
    ])

    dataset = Subset(
        torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform
        ),
        range(NUM_IMAGES)
    )

    # ========================================================
    # Modelos
    # ========================================================

    reference = LeNet5(
        num_classes=NUM_CLASSES
    )

    reference.load_state_dict(
        torch.load(
            args.weights,
            map_location="cpu",
            weights_only=True
        )
    )

    reference.eval()

    model = FpgaLeNet5()

    model.load_pytorch_weights(
        args.weights
    )

    model.eval()

    # ========================================================
    # Registros
    # ========================================================

    batch_rows = []
    run_rows = []
    prediction_rows = []
    summary_rows = []

    layer_rows = []
    layer_run_rows = []
    layer_summary_rows = []

    with torch.no_grad():

        for batch_size in args.batch_sizes:

            print(
                f"\n========== Batch {batch_size} =========="
            )

            # Datos preparados fuera de la medición.
            batches = list(
                DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False
                )
            )

            # Referencia calculada fuera de la medición FPGA.
            reference_outputs = [
                reference(images)
                for images, _ in batches
            ]

            labels = torch.cat([
                batch_labels
                for _, batch_labels in batches
            ]).numpy()

            ref_logits = torch.cat(
                reference_outputs
            )

            ref_predictions = (
                ref_logits.argmax(1).numpy()
            )

            # Una matriz de referencia por batch.
            np.savetxt(
                out / f"confusion_pytorch_b{batch_size}.csv",
                confusion(labels, ref_predictions),
                fmt="%d",
                delimiter=","
            )

            # =================================================
            # Calentamiento
            # =================================================

            for _ in range(args.warmup):
                for images, _ in batches:
                    model(images)

            # =================================================
            # Repeticiones principales
            # =================================================

            first_predictions = None
            configuration_runs = []

            confusion_sum = np.zeros(
                (NUM_CLASSES, NUM_CLASSES),
                dtype=np.float64
            )

            for repeat in range(1, args.repeats + 1):

                outputs = []
                durations = []

                # Este intervalo excluye datos, referencia,
                # cálculo de métricas y escritura de archivos.
                loop_start = perf_counter()

                for images, _ in batches:

                    start = perf_counter()

                    output = model(images)

                    end = perf_counter()

                    durations.append(end - start)
                    outputs.append(output)

                loop_seconds = (
                    perf_counter() - loop_start
                )

                # Métricas fuera del intervalo medido.
                logits = torch.cat(outputs)

                if not torch.isfinite(logits).all():
                    raise RuntimeError(
                        "La FPGA produjo NaN o infinito"
                    )

                predictions = (
                    logits.argmax(1).numpy()
                )

                if first_predictions is None:
                    first_predictions = predictions.copy()

                difference = (
                    logits - ref_logits
                ).double()

                rmse = float(
                    difference.square().mean().sqrt()
                )

                max_error = float(
                    difference.abs().max()
                )

                inference_seconds = sum(durations)

                confusion_sum += confusion(
                    labels,
                    predictions
                )

                row = {
                    "batch_size": batch_size,
                    "repeat": repeat,
                    "images": NUM_IMAGES,
                    "sum_model_calls_ms": (
                        inference_seconds * 1000
                    ),
                    "evaluation_loop_ms": (
                        loop_seconds * 1000
                    ),
                    "mean_batch_ms": (
                        float(np.mean(durations)) * 1000
                    ),
                    "amortized_ms_per_image": (
                        inference_seconds * 1000 / NUM_IMAGES
                    ),
                    "throughput_images_s": (
                        NUM_IMAGES / inference_seconds
                    ),
                    "accuracy_fpga_percent": (
                        float(np.mean(predictions == labels)) * 100
                    ),
                    "accuracy_pytorch_percent": (
                        float(np.mean(ref_predictions == labels)) * 100
                    ),
                    "agreement_count": int(
                        np.sum(predictions == ref_predictions)
                    ),
                    "changed_vs_repeat1": int(
                        np.sum(predictions != first_predictions)
                    ),
                    "rmse": rmse,
                    "max_error": max_error,
                    "rmse_pass": rmse <= TOLERANCE,
                }

                run_rows.append(row)
                configuration_runs.append(row)

                for index, seconds in enumerate(durations, 1):
                    batch_rows.append({
                        "batch_size": batch_size,
                        "repeat": repeat,
                        "batch_index": index,
                        "images": int(
                            batches[index - 1][0].shape[0]
                        ),
                        "model_call_ms": seconds * 1000,
                    })

                for index in range(NUM_IMAGES):
                    prediction_rows.append({
                        "batch_size": batch_size,
                        "repeat": repeat,
                        "mnist_index": index,
                        "label": int(labels[index]),
                        "pytorch": int(ref_predictions[index]),
                        "fpga": int(predictions[index]),
                    })

                # Conserva registros individuales para revisión.
                write_csv(
                    out / "runs.csv",
                    run_rows
                )

                write_csv(
                    out / "batches.csv",
                    batch_rows
                )

                write_csv(
                    out / "predictions.csv",
                    prediction_rows
                )

                print(
                    f"Repetición {repeat}/{args.repeats}: "
                    f"{inference_seconds * 1000:.3f} ms "
                    f"/ {NUM_IMAGES} imágenes | "
                    f"{NUM_IMAGES / inference_seconds:.3f} imágenes/s | "
                    f"RMSE {rmse:.3e}"
                )

            # =================================================
            # Matriz promedio
            # =================================================

            confusion_mean = (
                confusion_sum / args.repeats
            )

            np.savetxt(
                out / f"confusion_fpga_b{batch_size}_mean.csv",
                confusion_mean,
                fmt="%.4f",
                delimiter=","
            )

            # =================================================
            # Promedios de las repeticiones principales
            # =================================================

            summary = {
                "batch_size": batch_size,
                "repeats": args.repeats,
            }

            metrics = (
                "sum_model_calls_ms",
                "evaluation_loop_ms",
                "mean_batch_ms",
                "amortized_ms_per_image",
                "throughput_images_s",
                "accuracy_fpga_percent",
                "accuracy_pytorch_percent",
                "agreement_count",
                "changed_vs_repeat1",
                "rmse",
                "max_error",
            )

            for metric in metrics:
                values = [
                    row[metric]
                    for row in configuration_runs
                ]

                add_statistics(
                    summary,
                    metric,
                    values
                )

            total_seconds = (
                sum(
                    row["sum_model_calls_ms"]
                    for row in configuration_runs
                ) / 1000
            )

            summary["aggregate_throughput_images_s"] = (
                NUM_IMAGES * args.repeats / total_seconds
            )

            summary["accuracy_from_mean_confusion_percent"] = (
                100.0 * np.trace(confusion_mean) / NUM_IMAGES
            )

            # Distingue media de máximos del peor error observado.
            summary["max_error_worst"] = max(
                row["max_error"]
                for row in configuration_runs
            )

            summary["all_rmse_pass"] = all(
                row["rmse_pass"]
                for row in configuration_runs
            )

            summary_rows.append(summary)

            write_csv(
                out / "summary.csv",
                summary_rows
            )

            print(
                f"Promedio batch {batch_size}: "
                f"{summary['sum_model_calls_ms_mean']:.3f} "
                f"± {summary['sum_model_calls_ms_std']:.3f} ms "
                f"/ {NUM_IMAGES} imágenes"
            )

            # =================================================
            # Perfilado de capas en repeticiones separadas
            # =================================================

            if args.profile_layers:

                print(
                    f"Perfilado por capa: "
                    f"{args.repeats} pasadas adicionales"
                )

                starts = {}
                handles = []

                context = {
                    "repeat": 0,
                    "batch_index": 0,
                }

                layer_names = [
                    name
                    for name, _ in model.named_children()
                ]

                configuration_layer_runs = []

                def pre_hook(name):
                    def hook(module, inputs):
                        starts[name] = perf_counter()

                    return hook

                def post_hook(name):
                    def hook(module, inputs, output):
                        elapsed = (
                            perf_counter() - starts[name]
                        )

                        layer_rows.append({
                            "batch_size": batch_size,
                            "repeat": context["repeat"],
                            "batch_index": context["batch_index"],
                            "layer": name,
                            "python_layer_call_ms": elapsed * 1000,
                        })

                    return hook

                try:
                    for name, module in model.named_children():

                        handles.append(
                            module.register_forward_pre_hook(
                                pre_hook(name)
                            )
                        )

                        handles.append(
                            module.register_forward_hook(
                                post_hook(name)
                            )
                        )

                    for repeat in range(1, args.repeats + 1):

                        context["repeat"] = repeat
                        first_row = len(layer_rows)

                        for index, (images, _) in enumerate(
                            batches,
                            1
                        ):
                            context["batch_index"] = index
                            model(images)

                        repetition_rows = layer_rows[first_row:]

                        # Un resultado por capa para esta
                        # repetición de las 128 imágenes.
                        for name in layer_names:

                            values = [
                                row["python_layer_call_ms"]
                                for row in repetition_rows
                                if row["layer"] == name
                            ]

                            total_ms = float(np.sum(values))

                            layer_run = {
                                "batch_size": batch_size,
                                "repeat": repeat,
                                "layer": name,
                                "calls": len(values),
                                "sum_layer_calls_ms": total_ms,
                                "mean_layer_call_ms": float(
                                    np.mean(values)
                                ),
                                "amortized_layer_ms_per_image": (
                                    total_ms / NUM_IMAGES
                                ),
                            }

                            layer_run_rows.append(layer_run)
                            configuration_layer_runs.append(
                                layer_run
                            )

                        write_csv(
                            out / "layers_python_separate_pass.csv",
                            layer_rows
                        )

                        write_csv(
                            out / "layer_runs.csv",
                            layer_run_rows
                        )

                finally:
                    for handle in handles:
                        handle.remove()

                # Promedio entre las cinco repeticiones,
                # no mezcla indiscriminadamente los lotes.
                for name in layer_names:

                    selected = [
                        row
                        for row in configuration_layer_runs
                        if row["layer"] == name
                    ]

                    layer_summary = {
                        "batch_size": batch_size,
                        "layer": name,
                        "repeats": args.repeats,
                    }

                    for metric in (
                        "sum_layer_calls_ms",
                        "mean_layer_call_ms",
                        "amortized_layer_ms_per_image",
                    ):
                        add_statistics(
                            layer_summary,
                            metric,
                            [row[metric] for row in selected]
                        )

                    layer_summary_rows.append(
                        layer_summary
                    )

                write_csv(
                    out / "layer_summary.csv",
                    layer_summary_rows
                )

    print(f"\nResultados: {out}")

    print(
        "Matrices: filas = etiqueta real; "
        "columnas = predicción; clases 0 a 9."
    )

    print(
        "Pendientes: potencia, latencia delimitada "
        "por transferencias XRT y separación del overhead."
    )


if __name__ == "__main__":
    main()