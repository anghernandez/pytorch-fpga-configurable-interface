import argparse
import csv
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import threading
import psutil
import fpga_kernels
from models.FpgaMobileNetV2 import FpgaMobileNetV2
import torch
import torchvision
from PIL import Image
from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights,
)

# Reutiliza el mismo monitor y formato CSV de LeNet-5.
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




def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Benchmark MobileNetV2 CPU: PyTorch o C++."
    )

    parser.add_argument(
        "--implementation",
        choices=["fpga_baseline"],
        default="fpga_baseline",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=[1, 8, 16, 32, 64],
        default=1,
    )
    parser.add_argument("--images", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rest", type=float, default=5.0)
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "test_images/imagenet128",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=root.parent / "cpu_implementation/weights/mobilenetv2_imagenet1k_v1.pth",
    )

    args = parser.parse_args()

    if min(args.images, args.repeats, args.threads) < 1:
        parser.error("Imágenes, repeticiones e hilos deben ser positivos.")

    if args.rest < 0:
        parser.error("El descanso no puede ser negativo.")

    torch.set_num_threads(args.threads)

    data_dir = args.data.resolve()
    labels_path = data_dir / "labels.csv"
    weights_path = args.weights.resolve()

    # Leer etiquetas respetando el orden del CSV.
    with labels_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if not {"image", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("labels.csv debe contener image,label.")

        records = [
            {
                "image": row["image"],
                "label": int(row["label"]),
            }
            for row in reader
        ]

    if args.images > len(records):
        parser.error("La cantidad solicitada supera las imágenes del CSV.")

    records = records[:args.images]

    filenames = [record["image"] for record in records]

    if len(set(filenames)) != len(filenames):
        raise ValueError("Hay nombres de imagen duplicados.")

    for record in records:
        if not 0 <= record["label"] < 1000:
            raise ValueError("Las etiquetas deben estar entre 0 y 999.")

        if not (data_dir / record["image"]).is_file():
            raise FileNotFoundError(data_dir / record["image"])

    # Referencia oficial y preprocesamiento IMAGENET1K_V1.
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    if weights_path.exists():
        print("Cargando pesos locales:", weights_path, flush=True)

        reference = mobilenet_v2(weights=None)

        reference.load_state_dict(
            torch.load(
                weights_path,
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
    else:
        print(
            "Cargando pesos oficiales IMAGENET1K_V1 "
            "(se descargarán si no están en caché).",
            flush=True,
        )

        reference = mobilenet_v2(weights=weights)

        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(reference.state_dict(), weights_path)

        print("Pesos guardados:", weights_path, flush=True)

    reference = reference.cpu().eval()

    fpga_model = FpgaMobileNetV2(num_classes=1000)
    fpga_model.load_pytorch_weights(reference)
    fpga_model.eval()
    del reference

    # Desactivar el registro del perfilador por capas.
    fpga_kernels.profile_cancel()

    def model(batch):
        return fpga_model(batch, verbose=False)

    # Carga y preprocesamiento fuera del intervalo medido.
    print("Preparando imágenes...", flush=True)

    tensors = []
    manifest = []

    for index, record in enumerate(records):
        image_path = data_dir / record["image"]

        with Image.open(image_path) as image:
            tensor = preprocess(image.convert("RGB"))

        if tuple(tensor.shape) != (3, 224, 224):
            raise ValueError(
                "Forma de entrada inesperada: {}".format(tensor.shape)
            )

        tensors.append(tensor)

        manifest.append({
            "dataset_index": index,
            "image": record["image"],
            "true_label": record["label"],
            "sha256": sha256_file(image_path),
        })

    images = torch.stack(tensors)
    del tensors

    labels = torch.tensor(
        [record["label"] for record in records],
        dtype=torch.long,
    )

    batches = list(images.split(args.batch_size))

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    mode = "monitored" if args.monitor else "timing"

    output_dir = (
        root
        / "results"
        / "mobilenetv2_fpga_baseline"
        / "{}_b{}_{}".format(timestamp, args.batch_size, mode)
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "model": "MobileNetV2",
        "implementation": args.implementation,
        "device": "cpu",
        "mode": mode,
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "batch_size": args.batch_size,
        "images": args.images,
        "batches": len(batches),
        "dataset_directory": str(data_dir),
        "labels_csv_sha256": sha256_file(labels_path),
        "sample_order": "First rows of labels.csv, without shuffling",
        "preprocessing": str(preprocess),
        "weight_reference": "MobileNet_V2_Weights.IMAGENET1K_V1",
        "weights_path": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "repeats": args.repeats,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "warmup_full_passes": 1,
        "rest_seconds": args.rest,
        "timing_scope": (
            "Python loop and FPGA model calls, including interface "
            "management, transfers and kernel execution; "
            "preprocessed inputs already in CPU RAM"
        ),
        "power_and_resource_monitoring": args.monitor,
        "monitor_interval_seconds": 0.05 if args.monitor else None,
        "power_scope": "Kria SOM, ina260_u14",
        "energy_scope": "Monitor window surrounding timed inference",
        "cpu_scope": "Process including monitoring thread",
        "memory_scope": "Process RSS, sampled",
    }

    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    save_csv(
        output_dir / "images_manifest.csv",
        ["dataset_index", "image", "true_label", "sha256"],
        [
            [
                item["dataset_index"],
                item["image"],
                item["true_label"],
                item["sha256"],
            ]
            for item in manifest
        ],
    )

    categories = weights.meta["categories"]

    (output_dir / "classes.json").write_text(
        json.dumps(categories, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Implementación:", args.implementation, flush=True)
    print("Modo:", mode, flush=True)
    print("Imágenes:", args.images, flush=True)
    print("Batch:", args.batch_size, flush=True)
    print("Resultados:", output_dir, flush=True)
    print("Calentamiento: un recorrido completo.", flush=True)

    results = []
    first_predictions = None

    with torch.inference_mode():
        #for batch in batches:
            #warmup_output = model(batch)

        warmup_images = 0

        for batch_index, batch in enumerate(batches, start=1):
            print(
                f"Calentamiento: iniciando batch "
                f"{batch_index}/{len(batches)}",
                flush=True,
            )

            warmup_start = time.perf_counter()
            warmup_output = model(batch)
            warmup_elapsed = time.perf_counter() - warmup_start

            warmup_images += batch.shape[0]

            print(
                f"Calentamiento: {warmup_images}/{args.images} imágenes "
                f"| Último batch: {warmup_elapsed:.3f} s",
                flush=True,
            )

            if tuple(warmup_output.shape) != (batch.shape[0], 1000):
                raise RuntimeError("Forma de salida incorrecta.")

            if not torch.isfinite(warmup_output).all().item():
                raise RuntimeError("La salida contiene NaN o infinito.")

            del warmup_output

        print("Calentamiento terminado.", flush=True)

        for repetition in range(1, args.repeats + 1):
            time.sleep(args.rest)

            outputs = [None] * len(batches)
            monitor = None
            resources = {}

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

            # Métricas fuera del intervalo cronometrado.
            logits = torch.cat(outputs)

            if tuple(logits.shape) != (args.images, 1000):
                raise RuntimeError("Forma de salida incorrecta.")

            if not torch.isfinite(logits).all().item():
                raise RuntimeError("La salida contiene NaN o infinito.")

            predictions = logits.argmax(dim=1)

            correct = predictions.eq(labels).sum().item()
            accuracy = 100.0 * correct / args.images

            confusion = torch.bincount(
                labels * 1000 + predictions,
                minlength=1000 * 1000,
            ).reshape(1000, 1000)

            if first_predictions is None:
                first_predictions = predictions.clone()

            same = torch.equal(predictions, first_predictions)

            row = {
                "repetition": repetition,
                "implementation": args.implementation,
                "mode": mode,
                "batch_size": args.batch_size,
                "images": args.images,
                "batches": len(batches),
                "total_seconds": elapsed,
                "amortized_ms_per_image": elapsed * 1000 / args.images,
                "throughput_images_s": args.images / elapsed,
                "correct": correct,
                "accuracy_percent": accuracy,
                "same_predictions_as_first": same,
            }

            row.update(resources)
            results.append(row)

            save_csv(
                output_dir / "runs.csv",
                list(row),
                [list(result.values()) for result in results],
            )

            save_csv(
                output_dir / "predictions_rep{}.csv".format(repetition),
                ["dataset_index", "image", "true_label", "prediction"],
                [
                    [
                        index,
                        records[index]["image"],
                        labels[index].item(),
                        predictions[index].item(),
                    ]
                    for index in range(args.images)
                ],
            )

            save_csv(
                output_dir / "confusion_rep{}.csv".format(repetition),
                ["true_label"] + [
                    "pred_{}".format(index) for index in range(1000)
                ],
                (
                    [index] + confusion[index].tolist()
                    for index in range(1000)
                ),
            )

            if monitor is not None:
                first_time = monitor.samples[0][0]

                save_csv(
                    output_dir / "monitor_rep{}.csv".format(repetition),
                    ["elapsed_seconds", "power_som_w", "rss_mib"],
                    [
                        [sample_time - first_time, power, rss]
                        for sample_time, power, rss in monitor.samples
                    ],
                )

            print(
                "Rep {}/{} | {:.6f} s | {:.3f} ms/imagen | "
                "Accuracy: {:.2f}% | Iguales a rep. 1: {}".format(
                    repetition,
                    args.repeats,
                    elapsed,
                    row["amortized_ms_per_image"],
                    accuracy,
                    same,
                ),
                flush=True,
            )

            if monitor is not None:
                print(
                    "  SOM: {:.3f} W | Energía ventana: {:.3f} J | "
                    "CPU: {:.1f}% de capacidad total | "
                    "RSS máx.: {:.1f} MiB | Muestras: {}".format(
                        resources["power_som_mean_w"],
                        resources["energy_som_monitor_j"],
                        resources["cpu_process_machine_percent"],
                        resources["rss_peak_sampled_mib"],
                        resources["monitor_samples"],
                    ),
                    flush=True,
                )

            del logits, outputs, confusion

    summary = {}

    for metric, value in results[0].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue

        values = [result[metric] for result in results]

        summary[metric] = {
            "mean": statistics.mean(values),
            "std": (
                statistics.stdev(values) if len(values) > 1 else None
            ),
            "min": min(values),
            "max": max(values),
        }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(
        "\nTiempo medio: {:.6f} s".format(
            summary["total_seconds"]["mean"]
        )
    )
    print("Resultados guardados en:", output_dir)


if __name__ == "__main__":
    main()
