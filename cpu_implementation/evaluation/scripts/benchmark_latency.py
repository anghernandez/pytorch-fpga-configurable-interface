#!/usr/bin/env python3

"""
Benchmark de latencia de LeNet-5.

Compara:
    1. LeNet-5 de referencia en PyTorch.
    2. LeNet-5 implementada con kernels C++.

Plataforma:
    AMD Kria KV260 - CPU ARM Cortex-A53.

La medicion incluye:
    - Ejecucion completa del forward.
    - Overhead de wrappers y pybind11 en C++.
    - Generacion del tensor de salida.

La medicion excluye:
    - Carga de pesos.
    - Lectura del dataset.
    - Preprocesamiento de imagenes.
    - Construccion de los modelos.
"""

# ============================================================
# 1. CONFIGURACION DEL ENTORNO
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms


# ============================================================
# 2. DIRECTORIOS DEL PROYECTO
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

EVALUATION_DIR = SCRIPT_DIR.parent

PROJECT_DIR = EVALUATION_DIR.parent

PYTHON_DIR = PROJECT_DIR / "python"

WEIGHTS_PATH = (
    PROJECT_DIR / "weights" / "lenet5_reference.pth"
)

DATA_DIR = PROJECT_DIR / "data"

RESULTS_DIR = (
    EVALUATION_DIR / "results" / "latency"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Permitir importar los modelos y el modulo C++
sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(PROJECT_DIR))


# ============================================================
# 3. IMPORTAR LOS MODELOS EXISTENTES
# ============================================================

from models.LeNet5 import LeNet5
from models.CppLeNet5 import CppLeNet5


# ============================================================
# 4. CONFIGURACION EXPERIMENTAL
# ============================================================

NUM_WARMUP = 20

NUM_INFERENCES = 10000

BATCH_SIZE = 1

NUM_THREADS = 1

RMSE_THRESHOLD = 1e-3

torch.set_num_threads(NUM_THREADS)


# ============================================================
# 5. CARGAR DATASET MNIST
# ============================================================

def load_inputs():

    transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,)
        )
    ])

    dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        transform=transform,
        download=False
    )

    if len(dataset) < NUM_INFERENCES:
        raise ValueError(
            "El dataset contiene menos imagenes "
            "que las solicitadas."
        )

    inputs = []

    for index in range(NUM_INFERENCES):

        image, _ = dataset[index]

        input_tensor = (
            image.unsqueeze(0)
            .contiguous()
            .to(dtype=torch.float32)
        )

        inputs.append(input_tensor)

    return inputs


# ============================================================
# 6. CARGAR MODELOS
# ============================================================

def load_models():

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontraron los pesos: {WEIGHTS_PATH}"
        )

    # --------------------------------------------------------
    # Modelo PyTorch
    # --------------------------------------------------------

    pytorch_model = LeNet5(
        num_classes=10
    )

    state_dict = torch.load(
        str(WEIGHTS_PATH),
        map_location="cpu",
        weights_only=True
    )

    pytorch_model.load_state_dict(
        state_dict,
        strict=True
    )

    pytorch_model.eval()

    # --------------------------------------------------------
    # Modelo C++
    # --------------------------------------------------------

    cpp_model = CppLeNet5()

    cpp_model.load_pytorch_weights(
        str(WEIGHTS_PATH)
    )

    cpp_model.eval()

    return pytorch_model, cpp_model


# ============================================================
# 7. VALIDACION NUMERICA
# ============================================================

def validate_models(
    pytorch_model,
    cpp_model,
    input_tensor
):

    with torch.inference_mode():

        pytorch_output = pytorch_model(
            input_tensor
        )

        cpp_output = cpp_model(
            input_tensor
        )

    pytorch_array = (
        pytorch_output.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    cpp_array = (
        cpp_output.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    if pytorch_array.shape != cpp_array.shape:
        raise ValueError(
            "Las salidas tienen dimensiones diferentes."
        )

    difference = cpp_array - pytorch_array

    rmse = float(
        np.sqrt(
            np.mean(difference ** 2)
        )
    )

    max_error = float(
        np.max(np.abs(difference))
    )

    print("\n[VALIDACION NUMERICA]")

    print(f"RMSE:         {rmse:.10e}")
    print(f"Error maximo: {max_error:.10e}")

    if not np.isfinite(rmse):
        raise RuntimeError(
            "El RMSE no es finito."
        )

    if rmse > RMSE_THRESHOLD:
        raise RuntimeError(
            f"RMSE fuera del umbral: {rmse}"
        )

    print("Estado: PASS")

    return rmse


# ============================================================
# 8. MEDIR LATENCIA
# ============================================================

def measure_latency(
    model,
    inputs,
    model_name
):

    model.eval()

    latencies_ms = []

    print(
        f"\n[{model_name.upper()}] "
        "Ejecutando calentamiento..."
    )

    with torch.inference_mode():

        # ----------------------------------------------------
        # Warm-up
        # ----------------------------------------------------

        for index in range(NUM_WARMUP):

            input_tensor = inputs[
                index % len(inputs)
            ]

            model(input_tensor)

        # ----------------------------------------------------
        # Inferencias medidas
        # ----------------------------------------------------

        print(
            f"[{model_name.upper()}] "
            "Iniciando mediciones..."
        )

        for index in range(NUM_INFERENCES):

            input_tensor = inputs[index]

            start_ns = time.perf_counter_ns()

            output = model(input_tensor)

            end_ns = time.perf_counter_ns()

            latency_ms = (
                end_ns - start_ns
            ) / 1_000_000.0

            latencies_ms.append(
                latency_ms
            )

            if (index + 1) % 200 == 0:

                print(
                    f"[{model_name.upper()}] "
                    f"{index + 1}/{NUM_INFERENCES}"
                )

    return np.asarray(
        latencies_ms,
        dtype=np.float64
    )


# ============================================================
# 9. CALCULAR ESTADISTICAS
# ============================================================

def calculate_statistics(latencies_ms):

    return {
        "mean_ms": float(
            np.mean(latencies_ms)
        ),

        "median_ms": float(
            np.median(latencies_ms)
        ),

        "std_ms": float(
            np.std(latencies_ms, ddof=1)
        ),

        "p95_ms": float(
            np.percentile(latencies_ms, 95)
        ),

        "p99_ms": float(
            np.percentile(latencies_ms, 99)
        ),

        "min_ms": float(
            np.min(latencies_ms)
        ),

        "max_ms": float(
            np.max(latencies_ms)
        )
    }


# ============================================================
# 10. GUARDAR MEDICIONES CSV
# ============================================================

def save_latencies(
    model_name,
    latencies_ms
):

    output_path = (
        RESULTS_DIR /
        f"{model_name}_latency.csv"
    )

    with open(
        output_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "image_index",
            "latency_ms"
        ])

        for index, latency in enumerate(
            latencies_ms
        ):

            writer.writerow([
                index,
                f"{latency:.8f}"
            ])

    print(
        f"Archivo guardado: {output_path}"
    )


# ============================================================
# 11. GUARDAR RESUMEN CSV
# ============================================================

def save_summary(
    pytorch_stats,
    cpp_stats
):

    output_path = (
        RESULTS_DIR / "summary.csv"
    )

    fields = [
        "implementation",
        "num_inferences",
        "warmup",
        "batch_size",
        "mean_ms",
        "median_ms",
        "std_ms",
        "p95_ms",
        "p99_ms",
        "min_ms",
        "max_ms"
    ]

    with open(
        output_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for name, stats in [
            ("pytorch", pytorch_stats),
            ("cpp", cpp_stats)
        ]:

            row = {
                "implementation": name,
                "num_inferences": NUM_INFERENCES,
                "warmup": NUM_WARMUP,
                "batch_size": BATCH_SIZE
            }

            row.update(stats)

            writer.writerow(row)

    print(
        f"Resumen guardado: {output_path}"
    )


# ============================================================
# 12. MOSTRAR REPORTE
# ============================================================

def print_report(
    pytorch_stats,
    cpp_stats,
    rmse
):

    print("\n" + "=" * 70)

    print(
        "           BENCHMARK DE LATENCIA - LENET-5"
    )

    print(
        "                  AMD KRIA KV260"
    )

    print("=" * 70)

    print("\n[1] CONFIGURACION")

    print(f"Dataset:          MNIST Test")
    print(f"Batch size:       {BATCH_SIZE}")
    print(f"Warm-up:          {NUM_WARMUP}")
    print(f"Inferencias:      {NUM_INFERENCES}")
    print(f"Hilos PyTorch:    {torch.get_num_threads()}")
    print(f"Precision:        FP32")

    print("\n[2] LATENCIA DE INFERENCIA")

    print("-" * 70)

    print(
        f"{'Metrica':<24}"
        f"{'PyTorch (ms)':>20}"
        f"{'C++ (ms)':>20}"
    )

    print("-" * 70)

    metrics = [
        ("Media", "mean_ms"),
        ("Mediana", "median_ms"),
        ("Desv. estandar", "std_ms"),
        ("P95", "p95_ms"),
        ("P99", "p99_ms"),
        ("Minimo", "min_ms"),
        ("Maximo", "max_ms")
    ]

    for label, key in metrics:

        print(
            f"{label:<24}"
            f"{pytorch_stats[key]:>20.6f}"
            f"{cpp_stats[key]:>20.6f}"
        )

    print("-" * 70)

    # --------------------------------------------------------
    # Speedup
    # --------------------------------------------------------

    speedup = (
        pytorch_stats["mean_ms"]
        / cpp_stats["mean_ms"]
    )

    # --------------------------------------------------------
    # Throughput estimado
    # --------------------------------------------------------

    throughput_pytorch = (
        1000.0 / pytorch_stats["mean_ms"]
    )

    throughput_cpp = (
        1000.0 / cpp_stats["mean_ms"]
    )

    print("\n[3] COMPARACION")

    print(
        f"Speedup C++ / PyTorch: "
        f"{speedup:.4f}x"
    )

    print(
        f"Throughput PyTorch:    "
        f"{throughput_pytorch:.4f} inferencias/s"
    )

    print(
        f"Throughput C++:        "
        f"{throughput_cpp:.4f} inferencias/s"
    )

    print("\n[4] VALIDACION NUMERICA")

    print(f"RMSE:   {rmse:.10e}")
    print("Estado: PASS")

    print("\n" + "=" * 70)


# ============================================================
# 13. PROGRAMA PRINCIPAL
# ============================================================

def main():

    print("\nCargando imagenes MNIST...")

    inputs = load_inputs()

    print(
        f"Imagenes preparadas: {len(inputs)}"
    )

    print("\nCargando modelos...")

    pytorch_model, cpp_model = load_models()

    # --------------------------------------------------------
    # Validacion antes del benchmark
    # --------------------------------------------------------

    rmse = validate_models(
        pytorch_model,
        cpp_model,
        inputs[0]
    )

    # --------------------------------------------------------
    # Benchmark PyTorch
    # --------------------------------------------------------

    pytorch_latencies = measure_latency(
        pytorch_model,
        inputs,
        "pytorch"
    )

    save_latencies(
        "pytorch",
        pytorch_latencies
    )

    # --------------------------------------------------------
    # Benchmark C++
    # --------------------------------------------------------

    cpp_latencies = measure_latency(
        cpp_model,
        inputs,
        "cpp"
    )

    save_latencies(
        "cpp",
        cpp_latencies
    )

    # --------------------------------------------------------
    # Estadisticas
    # --------------------------------------------------------

    pytorch_stats = calculate_statistics(
        pytorch_latencies
    )

    cpp_stats = calculate_statistics(
        cpp_latencies
    )

    # --------------------------------------------------------
    # Guardar resumen
    # --------------------------------------------------------

    save_summary(
        pytorch_stats,
        cpp_stats
    )

    # --------------------------------------------------------
    # Reporte final
    # --------------------------------------------------------

    print_report(
        pytorch_stats,
        cpp_stats,
        rmse
    )


if __name__ == "__main__":
    main()
