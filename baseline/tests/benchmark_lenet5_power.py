"""LeNet-5 baseline: 128 imágenes, tiempos, potencia SOM y validación.

Potencia en proceso separado (spawn).

"""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import platform
import time
from datetime import datetime, timezone
from pathlib import Path


def write_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def sensor_worker(path, interval, stop, pipe):
    # No importa PyTorch: proceso de muestreo pequeño e independiente del GIL.
    samples = []
    try:
        while True:
            t0 = time.perf_counter()
            watts = int(Path(path).read_text()) / 1_000_000
            t1 = time.perf_counter()
            samples.append(((t0 + t1) / 2, watts, t1 - t0))
            if len(samples) == 1:
                pipe.send(('ready', None))
            if stop.wait(max(0.0, interval - (time.perf_counter() - t0))):
                break
        pipe.send(('samples', samples))
    except Exception as exc:
        pipe.send(('error', repr(exc)))
    finally:
        pipe.close()


class PowerRecorder:
    def __init__(self, path, interval):
        self.interval = interval
        ctx = mp.get_context('spawn')
        self.stop_event = ctx.Event()
        self.parent, child = ctx.Pipe(duplex=False)
        self.process = ctx.Process(
            target=sensor_worker,
            args=(str(path), interval, self.stop_event, child)
        )
        self.process.start()
        child.close()
        if not self.parent.poll(30):
            self.abort()
            raise RuntimeError('El monitor de potencia no respondió')
        kind, payload = self.parent.recv()
        if kind != 'ready':
            self.abort()
            raise RuntimeError(f'Error del sensor: {payload}')

    def finish(self):
        # Muestra posterior para interpolar el límite final, fuera del cronómetro.
        time.sleep(2 * self.interval)
        self.stop_event.set()
        try:
            if not self.parent.poll(15):
                raise RuntimeError('No se recibieron las muestras de potencia')
            kind, payload = self.parent.recv()
            if kind != 'samples':
                raise RuntimeError(f'Error del sensor: {payload}')
            return payload
        finally:
            self.abort()

    def abort(self):
        self.stop_event.set()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        self.parent.close()


def power_statistics(samples, start, end):
    import numpy as np
    data = np.asarray(samples, dtype=float)
    times, watts = data[:, 0], data[:, 1]
    interior = (times >= start) & (times <= end)
    if interior.sum() < 2 or times[0] > start or times[-1] < end:
        raise RuntimeError('Muestreo insuficiente para cubrir la evaluación')
    # Promedio temporal: integral trapezoidal / duración.
    t = np.concatenate(([start], times[(times > start) & (times < end)], [end]))
    p = np.interp(t, times, watts)
    energy = float(np.sum((p[1:] + p[:-1]) * 0.5 * np.diff(t)))
    return {
        'power_mean_w': energy / (end - start),
        'power_sample_max_w': float(watts[interior].max()),
        'power_samples': int(interior.sum()),
        'power_sample_interval_mean_ms': float(np.diff(times[interior]).mean() * 1000),
        'power_window_ms': (end - start) * 1000,
        'energy_estimate_j': energy,
    }


def main():
    import numpy as np
    import torch
    import torchvision
    from torchvision import transforms
    from torch.utils.data import DataLoader, Subset
    from models.LeNet5_reference import LeNet5
    from models.fpgaLeNet5 import FpgaLeNet5

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-sizes', nargs='+', type=int,
                        choices=[1, 8, 32, 64, 128], default=[1, 8, 32, 64, 128])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--idle-seconds', type=float, default=5)
    parser.add_argument('--sample-ms', type=float, default=50)
    parser.add_argument('--weights', default='weights/lenet5_reference.pth')
    parser.add_argument('--output', default='results/lenet5_power')
    parser.add_argument('--profile-layers', action='store_true')
    args = parser.parse_args()
    if args.repeats < 2 or args.warmup < 1 or args.sample_ms <= 0 or args.idle_seconds <= 0:
        parser.error('Repeticiones >= 2, calentamiento >= 1 y tiempos positivos')
    args.batch_sizes = list(dict.fromkeys(args.batch_sizes))
    matches = [d for d in Path('/sys/class/hwmon').glob('hwmon*')
               if (d / 'name').read_text().strip() == 'ina260_u14']
    if len(matches) != 1:
        raise RuntimeError(f'Se esperaba un sensor ina260_u14; encontrados: {matches}')
    sensor = matches[0] / 'power1_input'
    int(sensor.read_text())  # Comprueba permisos antes de cargar modelos.
    interval = args.sample_ms / 1000
    out = Path(args.output) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    metadata = {
        'arguments': vars(args), 'image_indices': list(range(128)),
        'dataset': 'MNIST test, primeras 128 imágenes, sin shuffle',
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'platform': platform.platform(), 'torch_threads': torch.get_num_threads(),
        'weights_sha256': hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
        'sensor': str(sensor), 'sensor_name': 'ina260_u14', 'sensor_unit': 'microWatt',
        'power_scope': 'Alimentación VCC_SOM; no exclusivamente lógica FPGA',
        'power_mean_method': 'Integral trapezoidal con límites interpolados / duración',
        'power_window': 'Bucle FPGA de 128 imágenes, sin métricas ni referencia PyTorch',
        'idle': 'Una ventana por batch después del calentamiento; monitor activo',
        'timing': 'Llamadas Python síncronas; monitor de potencia activo',
        'layer_timing': 'Repeticiones separadas, sin monitor de potencia',
        'std_ddof': 1, 'xrt_transfer_latency_measured': False,
        'overhead_separated': False,
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    dataset = Subset(torchvision.datasets.MNIST(
        './data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
        ])
    ), range(128))
    reference = LeNet5(num_classes=10)
    reference.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
    reference.eval()
    model = FpgaLeNet5()
    model.load_pytorch_weights(args.weights)
    model.eval()
    runs, batches_log, predictions_log, summaries = [], [], [], []
    raw_power, idle_log, layer_raw, layer_runs, layer_summaries = [], [], [], [], []

    def matrix(labels, pred):
        return np.bincount(10 * labels + pred, minlength=100).reshape(10, 10)

    def add_stats(dest, key, values):
        dest[key + '_mean'] = float(np.mean(values))
        dest[key + '_std'] = float(np.std(values, ddof=1))

    def record_samples(samples, batch_size, repeat, phase, start, end):
        for stamp, watts, read_s in samples:
            raw_power.append({
                'batch_size': batch_size, 'repeat': repeat, 'phase': phase,
                'monotonic_s': stamp, 'relative_s': stamp - start,
                'inside_window': start <= stamp <= end,
                'power_w': watts, 'read_duration_ms': read_s * 1000,
            })
        write_csv(out / 'power_samples.csv', raw_power)

    with torch.no_grad():
        for b in args.batch_sizes:
            data = list(DataLoader(dataset, batch_size=b, shuffle=False))
            labels = torch.cat([y for _, y in data]).numpy()
            ref = torch.cat([reference(x) for x, _ in data])
            ref_pred = ref.argmax(1).numpy()
            np.savetxt(out / f'confusion_pytorch_b{b}.csv', matrix(labels, ref_pred),
                       fmt='%d', delimiter=',')
            for _ in range(args.warmup):
                for x, _ in data:
                    model(x)
            print(f'Batch {b}: midiendo reposo durante {args.idle_seconds:g} s', flush=True)
            monitor = PowerRecorder(sensor, interval)
            try:
                start = time.perf_counter()
                time.sleep(args.idle_seconds)
                end = time.perf_counter()
            except BaseException:
                monitor.abort()
                raise
            samples = monitor.finish()
            record_samples(samples, b, 0, 'idle', start, end)
            idle = power_statistics(samples, start, end)
            idle_log.append({'batch_size': b, **idle})
            write_csv(out / 'idle_power.csv', idle_log)
            local_runs, confusion_sum, first_pred = [], np.zeros((10, 10)), None
            for repeat in range(1, args.repeats + 1):
                outputs, durations = [], []
                monitor = PowerRecorder(sensor, interval)
                try:
                    start = time.perf_counter()
                    for x, _ in data:
                        t0 = time.perf_counter()
                        output = model(x)
                        t1 = time.perf_counter()
                        outputs.append(output)
                        durations.append(t1 - t0)
                    end = time.perf_counter()
                except BaseException:
                    monitor.abort()
                    raise
                samples = monitor.finish()
                record_samples(samples, b, repeat, 'inference', start, end)
                power = power_statistics(samples, start, end)
                output = torch.cat(outputs)
                if not torch.isfinite(output).all():
                    raise RuntimeError('Salida FPGA no finita')
                pred = output.argmax(1).numpy()
                if first_pred is None:
                    first_pred = pred.copy()
                diff = (output - ref).double()
                rmse = float(diff.square().mean().sqrt())
                confusion_sum += matrix(labels, pred)
                seconds = sum(durations)
                row = {
                    'batch_size': b, 'repeat': repeat, 'images': 128,
                    'sum_model_calls_ms': seconds * 1000,
                    'evaluation_loop_ms': (end - start) * 1000,
                    'mean_batch_ms': float(np.mean(durations)) * 1000,
                    'amortized_ms_per_image': seconds * 1000 / 128,
                    'throughput_images_s': 128 / seconds,
                    'accuracy_fpga_percent': float(np.mean(pred == labels)) * 100,
                    'accuracy_pytorch_percent': float(np.mean(ref_pred == labels)) * 100,
                    'agreement_count': int(np.sum(pred == ref_pred)),
                    'changed_vs_repeat1': int(np.sum(pred != first_pred)),
                    'rmse': rmse, 'max_error': float(diff.abs().max()),
                    'rmse_pass': rmse <= 1e-3, **power,
                    'idle_power_mean_w': idle['power_mean_w'],
                    'power_increment_vs_idle_w': power['power_mean_w'] - idle['power_mean_w'],
                    'energy_estimate_j_per_image': power['energy_estimate_j'] / 128,
                }
                runs.append(row)
                local_runs.append(row)
                for idx, dt in enumerate(durations, 1):
                    batches_log.append({'batch_size': b, 'repeat': repeat, 'batch_index': idx,
                                        'images': b, 'model_call_ms': dt * 1000})
                for idx in range(128):
                    predictions_log.append({'batch_size': b, 'repeat': repeat,
                                            'mnist_index': idx, 'label': int(labels[idx]),
                                            'pytorch': int(ref_pred[idx]), 'fpga': int(pred[idx])})
                write_csv(out / 'runs.csv', runs)
                write_csv(out / 'batches.csv', batches_log)
                write_csv(out / 'predictions.csv', predictions_log)
                print(f'Batch {b}, repetición {repeat}: {seconds * 1000:.2f} ms; '
                      f'{power["power_mean_w"]:.3f} W SOM; RMSE {rmse:.3e}', flush=True)
            cm = confusion_sum / args.repeats
            np.savetxt(out / f'confusion_fpga_b{b}_mean.csv', cm, fmt='%.4f', delimiter=',')
            summary = {'batch_size': b, 'repeats': args.repeats}
            metrics = [k for k in local_runs[0] if k not in
                       ('batch_size', 'repeat', 'images', 'rmse_pass', 'idle_power_mean_w')]
            for key in metrics:
                add_stats(summary, key, [r[key] for r in local_runs])
            summary.update({
                'idle_power_mean_w': idle['power_mean_w'],
                'aggregate_throughput_images_s': 128 * args.repeats /
                    (sum(r['sum_model_calls_ms'] for r in local_runs) / 1000),
                'accuracy_from_mean_confusion_percent': float(np.trace(cm) / 128 * 100),
                'max_error_worst': max(r['max_error'] for r in local_runs),
                'all_rmse_pass': all(r['rmse_pass'] for r in local_runs),
            })
            summaries.append(summary)
            write_csv(out / 'summary.csv', summaries)

            if args.profile_layers:
                print(f'Batch {b}: {args.repeats} pasadas de capas, sin monitor', flush=True)
                starts, handles, context, local_layers = {}, [], {}, []
                names = [name for name, _ in model.named_children()]
                def before(name):
                    def hook(module, inputs):
                        starts[name] = time.perf_counter()
                    return hook
                def after(name):
                    def hook(module, inputs, result):
                        dt = time.perf_counter() - starts[name]
                        layer_raw.append({'batch_size': b, 'repeat': context['repeat'],
                                          'batch_index': context['index'], 'layer': name,
                                          'python_layer_call_ms': dt * 1000})
                    return hook
                try:
                    for name, module in model.named_children():
                        handles.append(module.register_forward_pre_hook(before(name)))
                        handles.append(module.register_forward_hook(after(name)))
                    for repeat in range(1, args.repeats + 1):
                        context['repeat'] = repeat
                        offset = len(layer_raw)
                        for idx, (x, _) in enumerate(data, 1):
                            context['index'] = idx
                            model(x)
                        for name in names:
                            values = [r['python_layer_call_ms'] for r in layer_raw[offset:]
                                      if r['layer'] == name]
                            row = {'batch_size': b, 'repeat': repeat, 'layer': name,
                                   'calls': len(values), 'sum_layer_calls_ms': float(np.sum(values)),
                                   'mean_layer_call_ms': float(np.mean(values)),
                                   'amortized_layer_ms_per_image': float(np.sum(values)) / 128}
                            layer_runs.append(row)
                            local_layers.append(row)
                        write_csv(out / 'layers_python_separate_pass.csv', layer_raw)
                        write_csv(out / 'layer_runs.csv', layer_runs)
                finally:
                    for h in handles:
                        h.remove()
                for name in names:
                    selected = [r for r in local_layers if r['layer'] == name]
                    row = {'batch_size': b, 'layer': name, 'repeats': args.repeats}
                    for key in ('sum_layer_calls_ms', 'mean_layer_call_ms',
                                'amortized_layer_ms_per_image'):
                        add_stats(row, key, [r[key] for r in selected])
                    layer_summaries.append(row)
                write_csv(out / 'layer_summary.csv', layer_summaries)
    print(f'Resultados: {out}')
    print('Potencia SOM, no exclusiva FPGA. Máximo muestreado, no pico eléctrico real.')
    print('Incremento frente al reposo no equivale a potencia exclusiva del kernel.')
    print('Pendientes: latencia de transferencias XRT y separación del overhead.')


if __name__ == '__main__':
    main()