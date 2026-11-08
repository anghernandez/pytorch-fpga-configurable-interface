import time
import numpy as np
import torch
import torch.nn.functional as F

import fpga_kernels


# ============================================================
# Modelo de prueba: Pointwise -> ReLU6 (4 bloques)
# ============================================================

class PointwiseBlock:
    def __init__(self, channels):
        self.weight = torch.randn(channels, channels, 1, 1) * 0.1
        self.bias = torch.randn(channels) * 0.05

        # Conversión a NumPy FP32 para la baseline
        self.weight_np = np.ascontiguousarray(
            self.weight.numpy(), dtype=np.float32
        )

        self.bias_np = np.ascontiguousarray(
            self.bias.numpy(), dtype=np.float32
        )


class TestModel:
    def __init__(self):
        self.blocks = [PointwiseBlock(8) for _ in range(4)]

    def baseline(self, x):
        pointwise_times = []
        relu6_times = []

        for block in self.blocks:

            # Pointwise FPGA
            start = time.perf_counter()

            x = fpga_kernels.pointwise_forward(
                x,
                block.weight_np,
                block.bias_np,
                True
            )

            end = time.perf_counter()

            pointwise_times.append(
                (end - start) * 1000
            )

            # ReLU6 FPGA
            start = time.perf_counter()

            x = fpga_kernels.relu6_forward(x)

            end = time.perf_counter()

            relu6_times.append(
                (end - start) * 1000
            )

        return x, pointwise_times, relu6_times

    def pytorch(self, x):
        for block in self.blocks:
            x = F.relu6(
                F.conv2d(x, block.weight, block.bias)
            )

        return x


# ============================================================
# Prueba de rendimiento
# ============================================================

def main():
    torch.manual_seed(42)

    batch_size = 64
    repeats = 5

    # Misma entrada que el DAG
    x = torch.randn(batch_size, 8, 28, 28)

    # Mismos pesos y sesgos que el DAG
    model = TestModel()

    x_np = np.ascontiguousarray(
        x.numpy(), dtype=np.float32
    )

    print("\n========== BASELINE ==========")
    print(f"Entrada: {x_np.shape}")
    print(f"Batch: {batch_size}")
    print(f"Bloques: {len(model.blocks)}")
    print(f"Operaciones: {len(model.blocks) * 2}")

    # ========================================================
    # Calentamiento
    # ========================================================

    print("\nCalentamiento...")

    model.baseline(x_np)

    # ========================================================
    # Mediciones
    # ========================================================

    print("\n========== MEDICIONES ==========")

    times = []
    pointwise_all = []
    relu6_all = []

    output = None

    for i in range(repeats):

        start = time.perf_counter()

        output, pointwise_times, relu6_times = (
            model.baseline(x_np)
        )

        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000

        times.append(elapsed_ms)
        pointwise_all.append(pointwise_times)
        relu6_all.append(relu6_times)

        print(
            f"Repetición {i+1}: {elapsed_ms:.3f} ms"
        )

    # ========================================================
    # Resumen general
    # ========================================================

    print("\n========== RESUMEN ==========")

    print(f"Batch: {batch_size}")
    print(f"Operaciones: {len(model.blocks) * 2}")
    print(f"Repeticiones: {repeats}")

    print(
        f"Latencia media: {np.mean(times):.3f} ms"
    )

    print(
        f"Latencia mínima: {np.min(times):.3f} ms"
    )

    print(
        f"Latencia máxima: {np.max(times):.3f} ms"
    )

    # ========================================================
    # Perfil por operación
    # ========================================================

    pointwise_all = np.asarray(pointwise_all)
    relu6_all = np.asarray(relu6_all)

    print("\n========== PERFIL POR OPERACIÓN ==========")

    for i in range(4):
        print(
            f"Pointwise {i+1}: "
            f"{np.mean(pointwise_all[:, i]):.3f} ms"
        )

        print(
            f"ReLU6 {i+1}: "
            f"{np.mean(relu6_all[:, i]):.3f} ms"
        )

    print("\n========== TIEMPOS ACUMULADOS ==========")

    print(
        f"Pointwise: "
        f"{np.mean(np.sum(pointwise_all, axis=1)):.3f} ms"
    )

    print(
        f"ReLU6: "
        f"{np.mean(np.sum(relu6_all, axis=1)):.3f} ms"
    )



    # ========================================================
    # Perfilado detallado XRT
    # ========================================================

    print("\n========== PERFIL DETALLADO XRT ==========")

    fpga_kernels.profile_begin(8)

    try:
        _, _, _ = model.baseline(x_np)
        profile = fpga_kernels.profile_end()
    except Exception:
        raise

    print("\nResultados del perfilador:")
    print(profile)
    # ========================================================
    # Validación contra PyTorch
    # ========================================================

    with torch.no_grad():
        reference = model.pytorch(x)

    output_torch = torch.from_numpy(
        np.array(output, copy=True)
    )

    if output_torch.shape != reference.shape:
        raise RuntimeError(
            f"Dimensiones incorrectas: "
            f"FPGA={tuple(output_torch.shape)}, "
            f"PyTorch={tuple(reference.shape)}"
        )

    diff = output_torch - reference

    rmse = torch.sqrt(
        torch.mean(diff ** 2)
    ).item()

    max_error = torch.max(
        torch.abs(diff)
    ).item()

    passed = (
        np.isfinite(rmse)
        and np.isfinite(max_error)
        and rmse <= 1e-3
        and max_error <= 1e-3
    )

    print("\n========== VALIDACIÓN ==========")

    print(f"RMSE vs PyTorch: {rmse:.8e}")
    print(f"Error máximo: {max_error:.8e}")

    print(
        f"Resultado: {'PASS' if passed else 'FAIL'}"
    )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
