"""Perfil C++/XRT de LeNet-5: 128 imágenes, cinco repeticiones por batch.
La latencia XRT es un intervalo del reloj del host entre llamadas sync.
No es tiempo exclusivo de hardware. No estima el coeficiente A del overhead.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import fpga_kernels
from models.LeNet5_reference import LeNet5
from models.fpgaLeNet5 import FpgaLeNet5

LAYERS = [
    ('conv1', 'Conv2D'), ('tanh1', 'Tanh'), ('pool1', 'AvgPool2D'),
    ('conv2', 'Conv2D'), ('tanh2', 'Tanh'), ('pool2', 'AvgPool2D'),
    ('fc1', 'Linear'), ('tanh3', 'Tanh'), ('fc2', 'Linear'),
    ('tanh4', 'Tanh'), ('fc3', 'Linear'),
]
METRICS = [
    'preparacion_ms', 'sync_hacia_fpga_ms', 'lanzamiento_ms', 'espera_ms',
    'lanzamiento_y_espera_ms', 'sync_hacia_cpu_ms', 'copia_salida_ms',
    'latencia_capa_ms', 'total_region_host_ms',
]


def save(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def stats(row, key, values):
    row[key + '_mean'] = float(np.mean(values))
    row[key + '_std'] = float(np.std(values, ddof=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-sizes', nargs='+', type=int, choices=[1, 8, 32, 64, 128],
                   default=[1, 8, 32, 64, 128])
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--weights', default='weights/lenet5_reference.pth')
    p.add_argument('--output', default='results/lenet5_xrt')
    a = p.parse_args()
    if a.repeats < 2 or a.warmup < 1:
        p.error('Usar al menos dos repeticiones y una pasada de calentamiento')
    for name in ('profile_begin', 'profile_end', 'profile_cancel'):
        if not hasattr(fpga_kernels, name):
            raise RuntimeError(f'Falta {name}; recompilá los bindings')
    a.batch_sizes = list(dict.fromkeys(a.batch_sizes))
    out = Path(a.output) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    metadata = {
        'arguments': vars(a), 'dataset': 'MNIST test', 'indices': list(range(128)),
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'module': fpga_kernels.__file__, 'torch_threads': torch.get_num_threads(),
        'weights_sha256': hashlib.sha256(Path(a.weights).read_bytes()).hexdigest(),
        'layers': LAYERS, 'std_ddof': 1,
        'latency': 'Reloj host: primera sync TO_DEVICE hasta última sync FROM_DEVICE por lote',
        'throughput': '128 imágenes / suma de latencias de los lotes de una repetición',
        'excluded': 'Calentamiento, referencia, datos, profile_begin/end, métricas y escritura',
        'instrumentation': 'Marcas de tiempo y append C++ incluidos; sin impresiones por capa',
        'power_measured': False, 'hardware_only_time_measured': False,
        'overhead_A_estimated': False,
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    dataset = Subset(torchvision.datasets.MNIST(
        './data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
        ])
    ), range(128))
    reference = LeNet5(num_classes=10)
    reference.load_state_dict(torch.load(a.weights, map_location='cpu', weights_only=True))
    reference.eval()
    model = FpgaLeNet5()
    model.load_pytorch_weights(a.weights)
    model.eval()
    all_batches, all_layers, all_runs, summaries = [], [], [], []
    layer_runs, layer_summaries, predictions = [], [], []
    fpga_kernels.profile_cancel()
    with torch.no_grad():
        for b in a.batch_sizes:
            data = list(DataLoader(dataset, batch_size=b, shuffle=False))
            labels = torch.cat([y for _, y in data]).numpy()
            ref = torch.cat([reference(x) for x, _ in data])
            ref_pred = ref.argmax(1).numpy()
            def confusion(pred):
                return np.bincount(10 * labels + pred, minlength=100).reshape(10, 10)
            np.savetxt(out / f'confusion_pytorch_b{b}.csv', confusion(ref_pred), fmt='%d', delimiter=',')
            for _ in range(a.warmup):
                for x, _ in data:
                    model(x)
            local_runs, local_layer_runs = [], []
            cm_sum, first_pred = np.zeros((10, 10)), None
            for repeat in range(1, a.repeats + 1):
                outputs, batch_rows, layer_rows = [], [], []
                for idx, (x, _) in enumerate(data, 1):
                    fpga_kernels.profile_begin(capacity=11)
                    try:
                        start = perf_counter()
                        output = model(x)
                        end = perf_counter()
                        profile = fpga_kernels.profile_end()
                    except BaseException:
                        fpga_kernels.profile_cancel()
                        raise
                    records = profile['layers']
                    if profile['operation_count'] != 11 or len(records) != 11:
                        raise RuntimeError('Se requieren exactamente 11 registros por lote')
                    if [r['operation'] for r in records] != [op for _, op in LAYERS]:
                        raise RuntimeError('Los nombres u orden de las capas no coinciden con LeNet-5')
                    latency = float(profile['latencia_red_ms'])
                    if not np.isfinite(latency) or latency <= 0:
                        raise RuntimeError('Latencia XRT inválida')
                    layer_sum = sum(r['latencia_capa_ms'] for r in records)
                    if layer_sum > latency + 1e-6:
                        raise RuntimeError('Los intervalos por capa exceden el intervalo de la red')
                    batch_rows.append({
                        'batch_size': b, 'repeat': repeat, 'batch_index': idx, 'images': len(x),
                        'model_call_ms': (end - start) * 1000, 'xrt_network_ms': latency,
                        'sum_layer_latency_ms': layer_sum,
                        'between_layer_intervals_ms': latency - layer_sum,
                        'outside_xrt_interval_ms': (end - start) * 1000 - latency,
                    })
                    for (name, operation), record in zip(LAYERS, records):
                        row = {'batch_size': b, 'repeat': repeat, 'batch_index': idx,
                               'layer': name, 'operation': operation}
                        row.update({key: float(record[key]) for key in METRICS})
                        if any(not np.isfinite(row[key]) or row[key] < 0 for key in METRICS):
                            raise RuntimeError(f'Medición inválida en {name}')
                        layer_rows.append(row)
                    outputs.append(output)
                logits = torch.cat(outputs)
                if not torch.isfinite(logits).all():
                    raise RuntimeError('Salida FPGA no finita')
                pred = logits.argmax(1).numpy()
                if first_pred is None:
                    first_pred = pred.copy()
                cm_sum += confusion(pred)
                diff = (logits - ref).double()
                total_xrt = sum(r['xrt_network_ms'] for r in batch_rows)
                total_python = sum(r['model_call_ms'] for r in batch_rows)
                row = {
                    'batch_size': b, 'repeat': repeat, 'images': 128,
                    'sum_model_calls_ms': total_python, 'sum_xrt_network_ms': total_xrt,
                    'mean_xrt_batch_ms': total_xrt / len(data),
                    'amortized_xrt_ms_per_image': total_xrt / 128,
                    'throughput_xrt_images_s': 128000 / total_xrt,
                    'throughput_python_images_s': 128000 / total_python,
                    'between_layer_intervals_ms': sum(r['between_layer_intervals_ms'] for r in batch_rows),
                    'outside_xrt_interval_ms': sum(r['outside_xrt_interval_ms'] for r in batch_rows),
                    'accuracy_fpga_percent': float(np.mean(pred == labels) * 100),
                    'accuracy_pytorch_percent': float(np.mean(ref_pred == labels) * 100),
                    'agreement_count': int(np.sum(pred == ref_pred)),
                    'changed_vs_repeat1': int(np.sum(pred != first_pred)),
                    'rmse': float(diff.square().mean().sqrt()), 'max_error': float(diff.abs().max()),
                }
                all_runs.append(row)
                local_runs.append(row)
                all_batches.extend(batch_rows)
                all_layers.extend(layer_rows)
                for name, op in LAYERS:
                    selected = [r for r in layer_rows if r['layer'] == name]
                    lr = {'batch_size': b, 'repeat': repeat, 'layer': name, 'operation': op}
                    for key in METRICS:
                        lr['sum_' + key] = sum(r[key] for r in selected)
                        lr['mean_call_' + key] = lr['sum_' + key] / len(selected)
                    layer_runs.append(lr)
                    local_layer_runs.append(lr)
                for idx in range(128):
                    predictions.append({'batch_size': b, 'repeat': repeat, 'mnist_index': idx,
                                        'label': int(labels[idx]), 'pytorch': int(ref_pred[idx]), 'fpga': int(pred[idx])})
                for filename, rows in [('runs.csv', all_runs), ('batches.csv', all_batches),
                                       ('layers.csv', all_layers), ('layer_runs.csv', layer_runs),
                                       ('predictions.csv', predictions)]:
                    save(out / filename, rows)
                print(f'Batch {b}, repetición {repeat}/{a.repeats}: suma latencias XRT '
                      f'{total_xrt:.3f} ms / 128 imágenes; RMSE {row["rmse"]:.3e}', flush=True)
            summary = {'batch_size': b, 'repeats': a.repeats}
            for key in local_runs[0]:
                if key not in ('batch_size', 'repeat', 'images'):
                    stats(summary, key, [r[key] for r in local_runs])
            summary['aggregate_throughput_xrt_images_s'] = 128000 * a.repeats / sum(r['sum_xrt_network_ms'] for r in local_runs)
            summary['all_rmse_pass'] = all(r['rmse'] <= 1e-3 for r in local_runs)
            summary['max_error_worst'] = max(r['max_error'] for r in local_runs)
            summaries.append(summary)
            save(out / 'summary.csv', summaries)
            np.savetxt(out / f'confusion_fpga_b{b}_mean.csv', cm_sum / a.repeats, fmt='%.4f', delimiter=',')
            for name, op in LAYERS:
                selected = [r for r in local_layer_runs if r['layer'] == name]
                row = {'batch_size': b, 'layer': name, 'operation': op, 'repeats': a.repeats}
                for key in selected[0]:
                    if key.startswith(('sum_', 'mean_call_')):
                        stats(row, key, [r[key] for r in selected])
                layer_summaries.append(row)
            save(out / 'layer_summary.csv', layer_summaries)
    print(f'Resultados: {out}')
    print('Los tiempos entre capas y fuera del intervalo XRT no equivalen a todo el overhead.')


if __name__ == '__main__':
    main()