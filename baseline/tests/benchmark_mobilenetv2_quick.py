import argparse
import csv
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "host")]

import torch
from PIL import Image
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from models.FpgaMobileNetV2 import FpgaMobileNetV2
import fpga_kernels

def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate(actual, expected):
    if actual.shape != expected.shape or not torch.isfinite(actual).all():
        raise RuntimeError("Salida FPGA inválida")
    diff = actual.double() - expected.double()
    per_image_rmse = diff.square().mean(1).sqrt()
    agreement = int((actual.argmax(1) == expected.argmax(1)).sum())
    return {
        "rmse": float(diff.square().mean().sqrt()),
        "max_error": float(diff.abs().max()),
        "max_rmse_per_image": float(per_image_rmse.max()),
        "agreement_count": agreement,
        "passed": (
            float(per_image_rmse.max()) <= 1e-3
            and agreement == expected.shape[0]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("results/mobilenetv2_quick"))
    args = parser.parse_args()

    started = perf_counter()
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    with (args.image_dir / "labels.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))[:8]

    if len(rows) != 8 or len({r["image"] for r in rows}) != 8:
        raise RuntimeError("Se necesitan ocho imágenes distintas")

    out = args.output / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    out.mkdir(parents=True, exist_ok=False)

    inputs, manifest = [], []
    for row in rows:
        path = args.image_dir / row["image"]
        with Image.open(path) as image:
            inputs.append(
                transform(image.convert("RGB")).unsqueeze(0).contiguous()
            )
        manifest.append({
            "image": row["image"],
            "label": int(row["label"]),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    write_csv(out / "image_manifest.csv", manifest)

    reference = mobilenet_v2(weights=weights).eval()
    model = FpgaMobileNetV2(num_classes=1000).eval()
    model.load_pytorch_weights(reference)

    runs, calls = [], []
    with torch.inference_mode():
        expected = torch.cat([reference(x) for x in inputs])

        fpga_kernels.profile_cancel()
        build_ms = 0.0  # No aplica: el baseline no construye un DAG. 
        print("Calentamiento: ocho imágenes...", flush=True)
        warm_outputs = [model(x, verbose=False) for x in inputs]
        warm_metrics = validate(torch.cat(warm_outputs), expected)
        del warm_outputs
        if not warm_metrics["passed"]:
            raise RuntimeError(f"Calentamiento no válido: {warm_metrics}")

        for repeat in range(1, 6):
            print(f"Repetición {repeat}/5...", flush=True)
            outputs, durations = [], []

            loop_start = perf_counter()
            for index, x in enumerate(inputs, 1):
                t0 = perf_counter()
                result = model(x, verbose=False)
                elapsed_ms = (perf_counter() - t0) * 1000
                outputs.append(result)
                durations.append(elapsed_ms)
            loop_ms = (perf_counter() - loop_start) * 1000

            # Validación y escritura fuera del intervalo medido.
            metrics = validate(torch.cat(outputs), expected)
            total_ms = sum(durations)
            runs.append({
                "repeat": repeat,
                "batch_size": 1,
                "images": 8,
                "sum_model_calls_ms": total_ms,
                "evaluation_loop_ms": loop_ms,
                "ms_per_image": total_ms / 8,
                "images_s": 8000 / total_ms,
                **metrics,
            })
            calls.extend({
                "repeat": repeat,
                "image": rows[i]["image"],
                "model_call_ms": duration,
            } for i, duration in enumerate(durations))

            write_csv(out / "runs.csv", runs)
            write_csv(out / "calls.csv", calls)
            print(
                f"  {total_ms / 1000:.3f} s | "
                f"RMSE={metrics['rmse']:.3e} | "
                f"PASS={metrics['passed']}",
                flush=True,
            )
            del outputs
            if not metrics["passed"]:
                raise RuntimeError("Validación fallida; revisar antes de continuar")

    times = [r["sum_model_calls_ms"] for r in runs]
    summary = {
        "batch_size": 1,
        "images": 8,
        "repeats": 5,
        "build_ms": build_ms,
        "sum_model_calls_ms_mean": statistics.mean(times),
        "sum_model_calls_ms_std": statistics.stdev(times),
        "ms_per_image_mean": statistics.mean(times) / 8,
        "aggregate_images_s": 40000 / sum(times),
        "all_passed": all(r["passed"] for r in runs),
    }
    write_csv(out / "summary.csv", [summary])
    metadata = {
        "weights": "IMAGENET1K_V1",
        "torch_threads": torch.get_num_threads(),
        "warmup_full_passes": 1,
        "power_monitor": False,
        "layer_profiling": False,
        "script_elapsed_s": perf_counter() - started,
        "script_elapsed_scope": (
            "Desde main: incluye preparación, referencia CPU, build, "
            "calentamiento, cinco repeticiones y validación; excluye imports"
        ),
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Media de ocho imágenes: {statistics.mean(times) / 1000:.3f} s "
        f"± {statistics.stdev(times) / 1000:.3f} s"
    )
    print(f"Resultados: {out}")


if __name__ == "__main__":
    main()