import argparse
import csv
import hashlib
import json
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
from torchvision import datasets, transforms

import fpga_kernels
from models.fpgaLeNet5 import FpgaLeNet5


def save_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


class ResourceMonitor:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.samples = []
        self.errors = []

        self.power_path = None

        for directory in Path("/sys/class/hwmon").glob("hwmon*"):
            name = directory / "name"

            if (
                name.exists()
                and name.read_text().strip() == "ina260_u14"
            ):
                candidate = directory / "power1_input"

                if candidate.exists():
                    self.power_path = candidate
                    break

        if self.power_path is None:
            raise RuntimeError("No se encontró el sensor ina260_u14.")

    def sample(self):
        power_w = float(
            self.power_path.read_text().strip()
        ) / 1_000_000

        rss_mib = self.process.memory_info().rss / (1024 ** 2)

        self.samples.append((
            time.perf_counter(),
            power_w,
            rss_mib
        ))

    def worker(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.sample()
            except Exception as error:
                self.errors.append(str(error))
                break

    def start(self):
        self.sample()
        self.cpu_start = self.process.cpu_times()
        self.wall_start = time.perf_counter()

        self.thread = threading.Thread(
            target=self.worker,
            daemon=True
        )
        self.thread.start()

    def stop(self):
        wall_end = time.perf_counter()
        cpu_end = self.process.cpu_times()

        self.stop_event.set()
        self.thread.join()
        self.sample()

        if self.errors:
            raise RuntimeError(
                "Error de monitoreo: " + "; ".join(self.errors)
            )

        duration = self.samples[-1][0] - self.samples[0][0]

        # Energía estimada sobre la ventana de monitoreo.
        energy_j = sum(
            (second[0] - first[0])
            * (first[1] + second[1]) / 2
            for first, second in zip(
                self.samples,
                self.samples[1:]
            )
        )

        cpu_seconds = (
            cpu_end.user + cpu_end.system
            - self.cpu_start.user - self.cpu_start.system
        )

        cpu_percent = (
            100 * cpu_seconds / (wall_end - self.wall_start)
        )
        logical_cpus = psutil.cpu_count(logical=True) or 1

        return {
            "monitor_window_seconds": duration,
            "power_som_mean_w": energy_j / duration,
            "power_som_peak_sampled_w": max(
                sample[1] for sample in self.samples
            ),
            "energy_som_monitor_j": energy_j,
            "cpu_process_percent": cpu_percent,
            "cpu_process_machine_percent": (
                cpu_percent / logical_cpus
            ),
            "rss_mean_sampled_mib": statistics.mean(
                sample[2] for sample in self.samples
            ),
            "rss_peak_sampled_mib": max(
                sample[2] for sample in self.samples
            ),
            "monitor_samples": len(self.samples)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación de LeNet-5 mediante la interfaz PyTorch-FPGA."
    )

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--images", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rest", type=float, default=5.0)

    parser.add_argument(
        "--implementation",
        choices=["fpga"],
        default="fpga"
    )

    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Activar monitoreo de potencia, CPU y memoria."
    )

    args = parser.parse_args()

    if min(
        args.batch_size,
        args.images,
        args.repeats,
        args.threads
    ) < 1:
        parser.error(
            "Batch, imágenes, repeticiones e hilos deben ser positivos."
        )

    if args.rest < 0:
        parser.error("El reposo no puede ser negativo.")

    root = Path(__file__).resolve().parents[1]
    weights_path = root / "weights/lenet5_reference.pth"

    torch.set_num_threads(args.threads)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        root=str(root / "data"),
        train=False,
        transform=transform,
        download=False
    )

    if args.images > len(dataset):
        parser.error(
            "La cantidad solicitada supera el conjunto de prueba."
        )

    # Preparación fuera de la medición.
    samples = [dataset[i] for i in range(args.images)]
    images = torch.stack([image for image, _ in samples])
    labels = torch.tensor([label for _, label in samples])
    batches = list(images.split(args.batch_size))

    # Desactivar el registro del perfilador personalizado.
    fpga_kernels.profile_cancel()

    model = FpgaLeNet5()
    model.load_pytorch_weights(str(weights_path))

    model.eval()

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    mode = "monitored" if args.monitor else "timing"

    output_dir = (
        root / "results" / "lenet5_fpga_baseline"
        / f"{timestamp}_b{args.batch_size}_{mode}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "model": "LeNet5",
        "implementation": args.implementation,
        "fpga_module": fpga_kernels.__file__,
        "custom_profile_recording": False,
        "mode": mode,
        "torch_version": str(torch.__version__),
        "batch_size": args.batch_size,
        "images": args.images,
        "dataset": "MNIST test",
        "indices": list(range(args.images)),
        "normalization_mean": 0.1307,
        "normalization_std": 0.3081,
        "repeats": args.repeats,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "warmup_full_passes": 1,
        "rest_seconds": args.rest,
        "weights_path": str(weights_path),
        "weights_sha256": hashlib.sha256(
            weights_path.read_bytes()
        ).hexdigest(),
        "timing_scope": (
            "FPGA model calls and Python loop; includes interface and transfers"
        ),
        "power_and_resource_monitoring": args.monitor,
        "monitor_interval_seconds": 0.05 if args.monitor else None,
        "power_scope": (
            "Kria SOM, sensor ina260_u14" if args.monitor else None
        ),
        "cpu_scope": (
            "Benchmark process, including monitoring thread"
            if args.monitor else None
        ),
        "memory_scope": (
            "Process RSS; sampled values" if args.monitor else None
        ),
        "energy_scope": (
            "Monitor window surrounding timed inference"
            if args.monitor else None
        )
    }

    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8"
    )

    results = []
    reference_predictions = None

    print(f"Implementación: {args.implementation}", flush=True)
    print(f"Modo: {mode}", flush=True)
    print(f"Resultados: {output_dir}", flush=True)
    print("Calentamiento: un recorrido completo.", flush=True)

    with torch.inference_mode():
        for batch in batches:
            model(batch)

        for repetition in range(1, args.repeats + 1):
            # Fuera de la medición.
            time.sleep(args.rest)
            outputs = [None] * len(batches)

            resources = {}
            monitor = None

            if args.monitor:
                monitor = ResourceMonitor(interval=0.05)
                monitor.start()

            try:
                start = time.perf_counter()

                for index, batch in enumerate(batches):
                    outputs[index] = model(batch)

                elapsed = time.perf_counter() - start
            finally:
                if monitor is not None:
                    resources = monitor.stop()

            # Evaluación y escritura fuera del intervalo cronometrado.
            predictions = torch.cat(outputs).argmax(dim=1)

            confusion = torch.bincount(
                labels * 10 + predictions,
                minlength=100
            ).reshape(10, 10)

            if reference_predictions is None:
                reference_predictions = predictions.clone()

            same_predictions = torch.equal(
                predictions,
                reference_predictions
            )

            accuracy = (
                100.0
                * predictions.eq(labels).sum().item()
                / args.images
            )

            row = {
                "repetition": repetition,
                "implementation": args.implementation,
                "mode": mode,
                "batch_size": args.batch_size,
                "images": args.images,
                "batches": len(batches),
                "total_seconds": elapsed,
                "amortized_ms_per_image": (
                    elapsed * 1000 / args.images
                ),
                "throughput_images_s": args.images / elapsed,
                "accuracy_percent": accuracy,
                "same_predictions_as_first": same_predictions
            }

            row.update(resources)
            results.append(row)

            # Conservar el progreso después de cada repetición.
            save_csv(
                output_dir / "runs.csv",
                list(row),
                [list(result.values()) for result in results]
            )

            save_csv(
                output_dir / f"predictions_rep{repetition}.csv",
                ["dataset_index", "true_label", "prediction"],
                zip(
                    range(args.images),
                    labels.tolist(),
                    predictions.tolist()
                )
            )

            save_csv(
                output_dir / f"confusion_rep{repetition}.csv",
                ["true_label"] + [f"pred_{i}" for i in range(10)],
                [
                    [i] + confusion[i].tolist()
                    for i in range(10)
                ]
            )

            if monitor is not None:
                # Conservar muestras para revisar el monitoreo.
                first_time = monitor.samples[0][0]

                save_csv(
                    output_dir / f"monitor_rep{repetition}.csv",
                    ["elapsed_seconds", "power_som_w", "rss_mib"],
                    [
                        [timestamp - first_time, power, memory]
                        for timestamp, power, memory in monitor.samples
                    ]
                )

            print(
                f"Rep {repetition}/{args.repeats} | "
                f"{elapsed:.6f} s | "
                f"{row['amortized_ms_per_image']:.3f} ms/imagen | "
                f"Accuracy: {accuracy:.2f}% | "
                f"Iguales a rep. 1: {same_predictions}",
                flush=True
            )

            if args.monitor:
                print(
                    f"  SOM: {resources['power_som_mean_w']:.3f} W | "
                    f"Energía: {resources['energy_som_monitor_j']:.3f} J | "
                    f"CPU: {resources['cpu_process_machine_percent']:.1f}% "
                    f"de la capacidad total | "
                    f"RAM máxima muestreada: "
                    f"{resources['rss_peak_sampled_mib']:.1f} MiB | "
                    f"Muestras: {resources['monitor_samples']}",
                    flush=True
                )

    summary = {}

    for metric in (
        "total_seconds",
        "amortized_ms_per_image",
        "throughput_images_s",
        "accuracy_percent",
        "monitor_window_seconds",
        "power_som_mean_w",
        "power_som_peak_sampled_w",
        "energy_som_monitor_j",
        "cpu_process_percent",
        "cpu_process_machine_percent",
        "rss_mean_sampled_mib",
        "rss_peak_sampled_mib",
        "monitor_samples"
    ):
        if metric not in results[0]:
            continue

        values = [result[metric] for result in results]

        summary[metric] = {
            "mean": statistics.mean(values),
            "std": (
                statistics.stdev(values)
                if len(values) > 1 else None
            ),
            "min": min(values),
            "max": max(values)
        }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print(
        "\nTiempo medio: "
        f"{summary['total_seconds']['mean']:.6f} s"
    )
    print(f"Resultados guardados en: {output_dir}")


if __name__ == "__main__":
    main()