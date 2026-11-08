"""MobileNetV2 baseline: 128 imágenes ImageNet-1K y cinco evaluaciones por batch.

Potencia SOM en proceso separado (spawn). Perfil XRT en pasadas separadas.
Ejecutar desde baseline con PYTHONPATH configurado. --limit permite una prueba corta.
"""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import platform
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def write_csv(path, rows):
    """Escritura atómica: nunca deja un CSV de resumen parcialmente escrito."""
    if not rows:
        return
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8',
                                     dir=path.parent, prefix=f'.{path.name}.',
                                     suffix='.tmp', delete=False) as f:
        temporary = Path(f.name)
        try:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def as_bool(value):
    return str(value).lower() in ('true', '1')


def write_json(path, payload):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=f'.{path.name}.', suffix='.tmp',
                                     delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write('\n')
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def resume_plan(folder, batch_sizes, repeats, n):
    """Verifica la consistencia de cada repetición guardada sin tocar los datos."""
    folder = Path(folder)
    original = folder / 'metadata.json'
    if not original.is_file():
        raise RuntimeError(f'No existe metadata.json en {folder}')
    meta = json.loads(original.read_text(encoding='utf-8'))
    if meta.get('images') != n or meta.get('arguments', {}).get('repeats') != repeats:
        raise RuntimeError('Las imágenes o repeticiones no coinciden con el experimento original')
    runs = read_csv(folder / 'runs.csv')
    profile_runs = read_csv(folder / 'profile_runs.csv')
    idle_rows = read_csv(folder / 'idle_power.csv')
    predictions = read_csv(folder / 'predictions.csv')
    batch_rows = read_csv(folder / 'batches.csv')
    profile_batches = read_csv(folder / 'profile_batches.csv')
    layer_runs = read_csv(folder / 'layer_runs.csv')
    seen = {'power': set(), 'profile': set(), 'idle': set()}
    for kind, table in (('power', runs), ('profile', profile_runs)):
        for row in table:
            key = int(row['batch_size']), int(row['repeat'])
            if key in seen[kind]:
                raise RuntimeError(f'Registro duplicado en {kind}: {key}')
            if not (1 <= key[1] <= repeats):
                raise RuntimeError(f'Repetición fuera de rango en {kind}: {key}')
            seen[kind].add(key)
    for row in idle_rows:
        key = int(row['batch_size'])
        if key in seen['idle']:
            raise RuntimeError(f'Registro idle duplicado para batch {key}')
        seen['idle'].add(key)
    for b in batch_sizes:
        if (folder / f'confusion_pytorch_b{b}.csv').exists() and not (folder / 'image_manifest.csv').exists():
            raise RuntimeError('Falta el manifiesto de imágenes')
        if b in seen['idle'] and not (folder / f'power_b{b}_r0_idle.csv').is_file():
            raise RuntimeError(f'Falta archivo de potencia idle batch {b}')
        if b not in seen['idle'] and (folder / f'power_b{b}_r0_idle.csv').exists():
            raise RuntimeError(f'Idle batch {b} parcialmente guardado; revisar manualmente')
        for r in range(1, repeats + 1):
            key = b, r
            pow_files = (folder / f'power_b{b}_r{r}_inference.csv',
                         folder / f'confusion_fpga_b{b}_r{r}.csv')
            if key in seen['power']:
                if not all(p.is_file() for p in pow_files):
                    raise RuntimeError(f'Archivos faltantes de potencia {key}')
                if len([x for x in predictions if (int(x['batch_size']), int(x['repeat'])) == key]) != n:
                    raise RuntimeError(f'Predicciones incompletas para {key}')
                if len([x for x in batch_rows if (int(x['batch_size']), int(x['repeat'])) == key]) != n // b:
                    raise RuntimeError(f'Lotes de potencia incompletos para {key}')
            elif any(p.exists() for p in pow_files):
                raise RuntimeError(f'Resultados de potencia incompletos para {key}; revisar antes de continuar')
            prof_file = folder / f'layers_xrt_b{b}_r{r}.csv'
            if key in seen['profile']:
                if not prof_file.is_file():
                    raise RuntimeError(f'Falta {prof_file.name}')
                if len([x for x in profile_batches if (int(x['batch_size']), int(x['repeat'])) == key]) != n // b:
                    raise RuntimeError(f'Lotes de perfil incompletos para {key}')
                if not any((int(x['batch_size']), int(x['repeat'])) == key for x in layer_runs):
                    raise RuntimeError(f'Faltan las estadísticas de capas para {key}')
            elif prof_file.exists():
                raise RuntimeError(f'Archivo de capas huérfano {prof_file.name}; revisar antes de continuar')
    return {
        b: {'idle_pending': b not in seen['idle'],
            'power_pending': [r for r in range(1, repeats + 1) if (b, r) not in seen['power']],
            'profile_pending': [r for r in range(1, repeats + 1) if (b, r) not in seen['profile']]}
        for b in batch_sizes
    }


def backup_aggregates(out):
    """Respalda los CSV agregados existentes antes de actualizar la corrida."""
    names = ['runs.csv', 'batches.csv', 'predictions.csv', 'idle_power.csv',
             'summary.csv', 'profile_runs.csv', 'profile_batches.csv',
             'layer_runs.csv', 'profile_summary.csv', 'layer_summary.csv']
    backup = out / '.respaldo_resume' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    backup.mkdir(parents=True, exist_ok=False)
    for name in names:
        if (out / name).exists():
            shutil.copy2(out / name, backup / name)
    return backup


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

    parser = argparse.ArgumentParser(
        description='MobileNetV2 baseline: potencia SOM y perfil XRT en pasadas separadas.'
    )
    parser.add_argument('--image-dir', default='test_images/imagenet128')
    parser.add_argument('--batch-sizes', nargs='+', type=int,
                        choices=[1, 8, 16, 32, 64], default=[1, 8, 16, 32, 64])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1,
                        help='Pasadas completas de calentamiento por configuración')
    parser.add_argument('--limit', type=int, default=128,
                        help='128 para el experimento; valores menores solo para comprobar el script')
    parser.add_argument('--idle-seconds', type=float, default=5)
    parser.add_argument('--sample-ms', type=float, default=50)
    parser.add_argument('--output', default='results/mobilenetv2_baseline')
    parser.add_argument('--resume-dir', type=Path,
                        help='Directorio existente con CSV de una corrida interrumpida')
    parser.add_argument('--dry-run', action='store_true',
                        help='Solo muestra el plan de reanudación; no ejecuta FPGA')
    args = parser.parse_args()
    if (args.repeats < 1 or args.warmup < 1 or args.limit < 1 or args.limit > 128
            or args.sample_ms <= 0 or args.idle_seconds <= 0):
        parser.error('Repeticiones y calentamiento >=1, limit entre 1 y 128, tiempos positivos')
    args.batch_sizes = list(dict.fromkeys(args.batch_sizes))
    if any(args.limit % b for b in args.batch_sizes):
        parser.error('--limit debe ser divisible por cada batch, sin batches parciales')
    if args.dry_run and not args.resume_dir:
        parser.error('--dry-run requiere --resume-dir')
    plan = None
    if args.resume_dir:
        plan = resume_plan(args.resume_dir, args.batch_sizes, args.repeats, args.limit)
        print('Plan de reanudación (no modifica mediciones existentes):', flush=True)
        for b, todo in plan.items():
            print(f'  Batch {b}: idle={"pendiente" if todo["idle_pending"] else "guardado"}; '
                  f'potencia pendientes={todo["power_pending"]}; '
                  f'perfil XRT pendientes={todo["profile_pending"]}', flush=True)
        if args.dry_run:
            return
    import gc
    from collections import Counter, defaultdict
    import numpy as np
    import torch
    import torchvision
    from PIL import Image
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    from models.FpgaMobileNetV2 import FpgaMobileNetV2
    import fpga_kernels

    matches = [d for d in Path('/sys/class/hwmon').glob('hwmon*')
               if (d / 'name').exists() and (d / 'name').read_text().strip() == 'ina260_u14']
    if len(matches) != 1:
        raise RuntimeError(f'Se esperaba un sensor ina260_u14; encontrados: {matches}')
    sensor = matches[0] / 'power1_input'
    int(sensor.read_text())
    interval = args.sample_ms / 1000
    folder = Path(args.image_dir)
    with (folder / 'labels.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))[:args.limit]
    n = len(rows)
    if n != args.limit or len({r['image'] for r in rows}) != n:
        raise RuntimeError('Cantidad de imágenes insuficiente o nombres repetidos')
    labels = np.asarray([int(r['label']) for r in rows], dtype=np.int64)
    if not ((labels >= 0) & (labels < 1000)).all():
        raise RuntimeError('Etiquetas fuera del rango 0..999')

    if args.resume_dir:
        out = args.resume_dir.resolve()
        if not out.is_dir():
            raise RuntimeError(f'No existe el directorio {out}')
    else:
        out = Path(args.output) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        out.mkdir(parents=True, exist_ok=False)
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    print(f'Preparando {n} imágenes; resultados: {out}', flush=True)
    inputs = []
    manifest = []
    for i, row in enumerate(rows):
        path = folder / row['image']
        with Image.open(path) as image:
            inputs.append(transform(image.convert('RGB')))
        manifest.append({'index': i, 'image': row['image'], 'label': int(labels[i]),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if args.resume_dir:
        saved_manifest = read_csv(out / 'image_manifest.csv')
        if saved_manifest != [{k: str(v) for k, v in row.items()} for row in manifest]:
            raise RuntimeError('Las imágenes, etiquetas u orden difieren del experimento original')
    else:
        write_csv(out / 'image_manifest.csv', manifest)

    # Solo se mantienen tensores individuales; no se duplica el conjunto completo con stack.
    def batches(b):
        return [torch.stack(inputs[i:i+b]) for i in range(0, n, b)]

    reference = mobilenet_v2(weights=weights).eval()
    model = FpgaMobileNetV2(num_classes=1000)
    model.load_pytorch_weights(reference)
    model.eval()
    state_hash = hashlib.sha256()
    for name, tensor in reference.state_dict().items():
        state_hash.update(name.encode())
        state_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())

    # Deriva el conteo esperado del modelo, excluyendo sumas residuales no ejecutadas.
    type_names = {
        'FpgaConv2D': 'Conv2D', 'FpgaPointwiseConv2d': 'Pointwise',
        'FpgaDepthwiseConv2d': 'Depthwise', 'FpgaBatchNorm2d': 'BatchNorm2D',
        'FpgaReLU6': 'ReLU6', 'FpgaLayerAdd': 'LayerAdd',
        'FpgaAvgPool2D': 'AvgPool2D', 'FpgaLinear': 'Linear',
    }
    inactive_adds = {id(block.add) for block in model.blocks if not block.use_residual}
    expected = Counter(type_names[type(m).__name__] for m in model.modules()
                       if type(m).__name__ in type_names and id(m) not in inactive_adds)
    if set(expected) != set(type_names.values()):
        raise RuntimeError(f'No se reconocieron todas las operaciones: {dict(expected)}')
    capacity = max(256, 2 * sum(expected.values()))
    print('Operaciones esperadas por llamada:', dict(expected), flush=True)

    metadata = {
        'arguments': vars(args), 'images': n,
        'experiment_complete_size': n == 128,
        'dataset': 'Primeras imágenes de ImageNet-1K validation descargadas previamente de ILSVRC/imagenet-1k',
        'weights': 'MobileNet_V2_Weights.IMAGENET1K_V1',
        'state_dict_sha256': state_hash.hexdigest(),
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'platform': platform.platform(), 'torch_threads': torch.get_num_threads(),
        'preprocess': str(transform), 'sensor': str(sensor), 'sensor_unit': 'microWatt',
        'power_scope': 'SOM completo; no potencia exclusiva de lógica FPGA',
        'power_window': 'Bucle de llamadas FPGA sin perfil activo; excluye referencia y métricas',
        'power_method': 'Integral trapezoidal con límites interpolados; máximo muestreado',
        'profile_window': 'Pasadas separadas sin monitor de potencia; una sesión por llamada',
        'xrt_latency': 'Antes del primer sync TO_DEVICE hasta después del último sync FROM_DEVICE',
        'launch_wait': 'Tiempo observado por CPU; no tiempo exclusivo de hardware',
        'unprofiled_regions': 'Bindings, Python, liberación de buffers y otras regiones fuera de marcas',
        'overhead_coefficient_estimated': False,
        'expected_operations_per_call': dict(expected),
        'profile_capacity': capacity, 'std_ddof': 1,
        'xclbin': '/lib/firmware/xilinx/fpga_accelerators/fpga_kernels.xclbin',
    }
    if args.resume_dir:
        saved_meta = json.loads((out / 'metadata.json').read_text(encoding='utf-8'))
        for key in ('state_dict_sha256', 'weights', 'xclbin', 'profile_capacity'):
            if saved_meta.get(key) != metadata.get(key):
                raise RuntimeError(f'Metadatos incompatibles para reanudar: {key}')
        backup = backup_aggregates(out)
        print(f'Respaldo de CSV agregados: {backup}', flush=True)
        write_json(out / 'resume_metadata.json', {
            'resumed_utc': datetime.now(timezone.utc).isoformat(),
            'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
            'source_metadata': 'metadata.json',
            'original_data_preserved': True,
        })
    else:
        write_json(out / 'metadata.json', metadata)

    def progress(phase, b=None, repeat=None, index=None, detail=None):
        write_json(out / 'resume_progress.json', {
            'utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
            'phase': phase, 'batch_size': b, 'repeat': repeat,
            'batch_index': index, 'detail': detail,
        })

    def matrix(pred):
        return np.bincount(1000 * labels + pred, minlength=1000000).reshape(1000, 1000)

    def stats(key, values):
        return {key + '_mean': float(np.mean([float(v) for v in values])),
                key + '_std': float(np.std([float(v) for v in values], ddof=1)) if len(values) > 1 else None}

    def validate(output, ref):
        if tuple(output.shape) != (n, 1000) or not torch.isfinite(output).all().item():
            raise RuntimeError('Salida FPGA incorrecta o no finita')
        diff = output.double() - ref.double()
        per_image = diff.square().mean(1).sqrt()
        pred = output.argmax(1).numpy()
        ref_pred = ref.argmax(1).numpy()
        metrics = {
            'accuracy_fpga_percent': float(np.mean(pred == labels) * 100),
            'accuracy_pytorch_percent': float(np.mean(ref_pred == labels) * 100),
            'agreement_count': int(np.sum(pred == ref_pred)),
            'agreement_percent': float(np.mean(pred == ref_pred) * 100),
            'rmse': float(diff.square().mean().sqrt()),
            'max_rmse_per_image': float(per_image.max()),
            'max_error': float(diff.abs().max()),
        }
        metrics['passed'] = metrics['max_rmse_per_image'] <= 1e-3 and metrics['agreement_count'] == n
        return metrics, pred, ref_pred

    def existing(name):
        return read_csv(out / name) if args.resume_dir else []

    runs, batch_rows, predictions = (existing(x) for x in ('runs.csv', 'batches.csv', 'predictions.csv'))
    idle_rows, summaries = (existing(x) for x in ('idle_power.csv', 'summary.csv'))
    profile_runs, profile_batches, layer_runs = (
        existing(x) for x in ('profile_runs.csv', 'profile_batches.csv', 'layer_runs.csv'))
    layer_summary, profile_summary = (existing(x) for x in ('layer_summary.csv', 'profile_summary.csv'))
    def put_summary(target, row):
        target[:] = [x for x in target if int(x['batch_size']) != int(row['batch_size'])]
        target.append(row)
        target.sort(key=lambda x: int(x['batch_size']))

    def has_repeat(table, b, repeat):
        return any(int(row['batch_size']) == b and int(row['repeat']) == repeat for row in table)

    phase_keys = ['preparacion_ms', 'sync_hacia_fpga_ms', 'lanzamiento_ms', 'espera_ms',
                  'lanzamiento_y_espera_ms', 'sync_hacia_cpu_ms', 'copia_salida_ms',
                  'latencia_capa_ms', 'total_region_host_ms']

    # Se escribe por ventana; las muestras no se acumulan en memoria entre repeticiones.
    def save_samples(samples, b, repeat, phase, start, end):
        write_csv(out / f'power_b{b}_r{repeat}_{phase}.csv', [
            {'batch_size': b, 'repeat': repeat, 'phase': phase, 'monotonic_s': stamp,
             'relative_s': stamp-start, 'inside_window': start <= stamp <= end,
             'power_w': watts, 'read_duration_ms': read_s*1000}
            for stamp, watts, read_s in samples])

    all_passed = True
    fpga_kernels.profile_cancel()
    with torch.inference_mode():
        for b in args.batch_sizes:
            data = batches(b)
            print(f'Batch {b}: referencia PyTorch', flush=True)
            ref = torch.cat([reference(x) for x in data])
            if tuple(ref.shape) != (n, 1000) or not torch.isfinite(ref).all().item():
                raise RuntimeError('Referencia no válida')
            if not (out / f'confusion_pytorch_b{b}.csv').exists():
                np.savetxt(out / f'confusion_pytorch_b{b}.csv', matrix(ref.argmax(1).numpy()),
                           fmt='%d', delimiter=',')
            for w in range(args.warmup):
                print(f'Batch {b}: calentamiento completo {w+1}/{args.warmup}', flush=True)
                for x in data:
                    warm = model(x, verbose=False)
                    del warm
            if has_repeat(runs, b, args.repeats) and len([r for r in runs if int(r['batch_size']) == b]) == args.repeats:
                print(f'Batch {b}: potencia completada previamente; se conserva', flush=True)
            else:
                print(f'Batch {b}: faltan mediciones de potencia', flush=True)
            progress('batch_iniciado', b)
            if not any(int(r['batch_size']) == b for r in idle_rows):
                print(f'Batch {b}: reposo {args.idle_seconds:g} s', flush=True)
                monitor = PowerRecorder(sensor, interval)
                try:
                    start = time.perf_counter()
                    time.sleep(args.idle_seconds)
                    end = time.perf_counter()
                    samples = monitor.finish()
                except BaseException:
                    monitor.abort()
                    raise
                idle = power_statistics(samples, start, end)
                save_samples(samples, b, 0, 'idle', start, end)
                del samples
                idle_rows.append({'batch_size': b, **idle})
                write_csv(out / 'idle_power.csv', idle_rows)
            else:
                idle = next(r for r in idle_rows if int(r['batch_size']) == b)
                idle = {k: float(v) if k != 'batch_size' else b for k, v in idle.items()}
            cm_sum = np.zeros((1000, 1000), dtype=np.int64)
            prior = [r for r in runs if int(r['batch_size']) == b]
            for prior_row in prior:
                saved_repeat = int(prior_row['repeat'])
                cm_sum += np.loadtxt(out / f'confusion_fpga_b{b}_r{saved_repeat}.csv',
                                     dtype=np.int64, delimiter=',')
            first_pred = None
            earlier_predictions = [r for r in predictions
                                   if int(r['batch_size']) == b and int(r['repeat']) == 1]
            if earlier_predictions:
                first_pred = np.array([int(r['fpga']) for r in earlier_predictions])
            local_runs = list(prior)
            for repeat in range(1, args.repeats+1):
                if has_repeat(runs, b, repeat):
                    print(f'Batch {b}: potencia/tiempo {repeat}/{args.repeats} ya guardada; OMITIDA', flush=True)
                    continue
                progress('potencia_iniciada', b, repeat)
                print(f'Batch {b}: potencia/tiempo, repetición {repeat}/{args.repeats}', flush=True)
                outputs, durations = [], []
                monitor = PowerRecorder(sensor, interval)
                try:
                    start = time.perf_counter()
                    for x in data:
                        t0 = time.perf_counter()
                        output = model(x, verbose=False)
                        t1 = time.perf_counter()
                        outputs.append(output)
                        durations.append(t1-t0)
                    end = time.perf_counter()
                    samples = monitor.finish()
                except BaseException:
                    monitor.abort()
                    raise
                power = power_statistics(samples, start, end)
                save_samples(samples, b, repeat, 'inference', start, end)
                del samples
                output = torch.cat(outputs)
                metrics, pred, ref_pred = validate(output, ref)
                all_passed = all_passed and metrics['passed']
                if first_pred is None:
                    first_pred = pred.copy()
                cm_sum += matrix(pred)
                np.savetxt(out / f'confusion_fpga_b{b}_r{repeat}.csv', matrix(pred),
                           fmt='%d', delimiter=',')
                seconds = sum(durations)
                row = {'batch_size': b, 'repeat': repeat, 'images': n,
                       'sum_model_calls_ms': seconds*1000,
                       'evaluation_loop_ms': (end-start)*1000,
                       'mean_batch_ms': seconds*1000/len(data),
                       'amortized_ms_per_image': seconds*1000/n,
                       'throughput_python_images_s': n/seconds,
                       'changed_vs_repeat1': int(np.sum(pred != first_pred)),
                       **metrics, **power,
                       'idle_power_mean_w': idle['power_mean_w'],
                       'power_increment_vs_idle_w': power['power_mean_w']-idle['power_mean_w'],
                       'energy_estimate_j_per_image': power['energy_estimate_j']/n,
                       'incremental_energy_vs_idle_j': power['energy_estimate_j']-idle['power_mean_w']*(end-start)}
                runs.append(row)
                local_runs.append(row)
                for i, dt in enumerate(durations, 1):
                    batch_rows.append({'batch_size': b, 'repeat': repeat, 'batch_index': i,
                                       'images': b, 'model_call_ms': dt*1000})
                for i in range(n):
                    predictions.append({'batch_size': b, 'repeat': repeat, 'image': rows[i]['image'],
                                        'label': int(labels[i]), 'pytorch': int(ref_pred[i]), 'fpga': int(pred[i])})
                write_csv(out / 'runs.csv', runs)
                write_csv(out / 'batches.csv', batch_rows)
                write_csv(out / 'predictions.csv', predictions)
                print(f'  {seconds:.3f} s; {power["power_mean_w"]:.3f} W SOM; '
                      f'RMSE {metrics["rmse"]:.3e}; PASS={metrics["passed"]}', flush=True)
                progress('potencia_guardada', b, repeat)
                del outputs, output
            cm_mean = cm_sum / args.repeats
            if not (out / f'confusion_fpga_b{b}_mean.csv').exists():
                np.savetxt(out / f'confusion_fpga_b{b}_mean.csv', cm_mean, fmt='%.6f', delimiter=',')
            summary = {'batch_size': b, 'repeats': args.repeats, 'images': n}
            for key in local_runs[0]:
                if key not in ('batch_size', 'repeat', 'images', 'passed'):
                    summary.update(stats(key, [r[key] for r in local_runs]))
            summary.update({'all_passed': all(as_bool(r['passed']) for r in local_runs),
                            'max_error_worst': max(float(r['max_error']) for r in local_runs),
                            'aggregate_throughput_python_images_s': n*args.repeats /
                            (sum(float(r['sum_model_calls_ms']) for r in local_runs)/1000),
                            'accuracy_from_mean_confusion_percent': float(np.trace(cm_mean)/n*100)})
            put_summary(summaries, summary)
            write_csv(out / 'summary.csv', summaries)

            print(f'Batch {b}: perfil XRT separado, sin monitor de potencia', flush=True)
            local_profile_runs = [r for r in profile_runs if int(r['batch_size']) == b]
            local_layer_runs = [r for r in layer_runs if int(r['batch_size']) == b]
            for repeat in range(1, args.repeats+1):
                if has_repeat(profile_runs, b, repeat):
                    print(f'  Perfil {repeat}/{args.repeats} ya guardado; OMITIDO', flush=True)
                    continue
                progress('perfil_iniciado', b, repeat)
                sum_python, sum_network, sum_layer_latency = 0.0, 0.0, 0.0
                grouped = defaultdict(lambda: defaultdict(float))
                counts = Counter()
                outputs = []
                raw_layers = []
                for index, x in enumerate(data, 1):
                    progress('inferencia_xrt_iniciada', b, repeat, index)
                    fpga_kernels.profile_begin(capacity)
                    try:
                        t0 = time.perf_counter()
                        result = model(x, verbose=False)
                        t1 = time.perf_counter()
                        profile = fpga_kernels.profile_end()
                    except BaseException:
                        fpga_kernels.profile_cancel()
                        raise
                    layers = profile['layers']
                    observed = Counter(layer['operation'] for layer in layers)
                    if observed != expected or profile['operation_count'] != sum(expected.values()):
                        raise RuntimeError(f'Perfil incompleto: esperado={dict(expected)}, observado={dict(observed)}')
                    python_ms = (t1-t0)*1000
                    network_ms = float(profile['latencia_red_ms'])
                    layer_latency_ms = sum(layer['latencia_capa_ms'] for layer in layers)
                    tolerance = max(0.01, python_ms*1e-6)
                    if not (0 <= layer_latency_ms <= network_ms+tolerance
                            and network_ms <= python_ms+tolerance):
                        raise RuntimeError('Marcas temporales inconsistentes')
                    sum_python += python_ms
                    sum_network += network_ms
                    sum_layer_latency += layer_latency_ms
                    profile_batches.append({'batch_size': b, 'repeat': repeat, 'batch_index': index,
                                            'images': b, 'model_call_ms': python_ms,
                                            'xrt_network_ms': network_ms,
                                            'between_layer_intervals_ms': network_ms-layer_latency_ms,
                                            'outside_xrt_interval_ms': python_ms-network_ms,
                                            'operation_count': len(layers)})
                    for operation_index, layer in enumerate(layers, 1):
                        if any(not np.isfinite(layer[k]) or layer[k] < 0 for k in phase_keys):
                            raise RuntimeError('Duración de etapa no válida')
                        raw_layers.append({'batch_size': b, 'repeat': repeat, 'batch_index': index,
                                           'operation_index': operation_index, **layer})
                        operation = layer['operation']
                        counts[operation] += 1
                        for key in phase_keys:
                            grouped[operation][key] += layer[key]
                    outputs.append(result)
                    progress('inferencia_xrt_terminada', b, repeat, index)
                metrics, _, _ = validate(torch.cat(outputs), ref)
                all_passed = all_passed and metrics['passed']
                row = {'batch_size': b, 'repeat': repeat, 'images': n,
                       'sum_model_calls_ms': sum_python, 'sum_xrt_network_ms': sum_network,
                       'mean_xrt_batch_ms': sum_network/len(data),
                       'amortized_xrt_ms_per_image': sum_network/n,
                       'throughput_xrt_images_s': n/(sum_network/1000),
                       'throughput_python_images_s': n/(sum_python/1000),
                       'between_layer_intervals_ms': sum_network-sum_layer_latency,
                       'outside_xrt_interval_ms': sum_python-sum_network, **metrics}
                profile_runs.append(row)
                local_profile_runs.append(row)
                for operation, values in sorted(grouped.items()):
                    layer_row = {'batch_size': b, 'repeat': repeat, 'operation': operation,
                                 'calls': counts[operation]}
                    for key, value in values.items():
                        layer_row['sum_'+key] = value
                        layer_row['mean_call_'+key] = value/counts[operation]
                    layer_runs.append(layer_row)
                    local_layer_runs.append(layer_row)
                write_csv(out / f'layers_xrt_b{b}_r{repeat}.csv', raw_layers)
                write_csv(out / 'profile_batches.csv', profile_batches)
                write_csv(out / 'profile_runs.csv', profile_runs)
                write_csv(out / 'layer_runs.csv', layer_runs)
                print(f'  Perfil {repeat}: red={sum_network/1000:.3f} s; '
                      f'PASS={metrics["passed"]}', flush=True)
                progress('perfil_guardado', b, repeat)
                del outputs, result, raw_layers
            profile_sum = {'batch_size': b, 'repeats': args.repeats, 'images': n}
            for key in local_profile_runs[0]:
                if key not in ('batch_size', 'repeat', 'images', 'passed'):
                    profile_sum.update(stats(key, [r[key] for r in local_profile_runs]))
            profile_sum['all_passed'] = all(as_bool(r['passed']) for r in local_profile_runs)
            put_summary(profile_summary, profile_sum)
            write_csv(out / 'profile_summary.csv', profile_summary)
            for operation in sorted(expected):
                selected = [r for r in local_layer_runs if r['operation'] == operation]
                row = {'batch_size': b, 'operation': operation, 'repeats': args.repeats}
                for key in selected[0]:
                    if key not in ('batch_size', 'repeat', 'operation'):
                        row.update(stats(key, [r[key] for r in selected]))
                layer_summary[:] = [x for x in layer_summary
                                    if not (int(x['batch_size']) == b and x['operation'] == operation)]
                layer_summary.append(row)
            layer_summary.sort(key=lambda x: (int(x['batch_size']), x['operation']))
            write_csv(out / 'layer_summary.csv', layer_summary)
            progress('batch_completado', b)
            del data, ref
            gc.collect()
    all_passed = all_passed and all(as_bool(r['passed']) for r in runs + profile_runs)
    write_json(out / 'status.json', {'completed': True, 'all_passed': all_passed,
                                      'resumed': bool(args.resume_dir),
                                      'batch_sizes_completed_this_run': args.batch_sizes})
    progress('finalizado')
    print(f'Resultados: {out}', flush=True)
    print('Potencia/energía: SOM completo. Perfil XRT: pasadas separadas; tiempos observados por CPU.')
    print('No se ha estimado el coeficiente de overhead del modelo del profesor.')
    if not all_passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()