import time
import torch
import torch.nn.functional as F

from fpga_graph import FpgaGraph


# ============================================================
# Modelo de prueba: Pointwise -> ReLU6 (4 bloques)
# ============================================================

class PointwiseBlock:
    def __init__(self, channels):
        self.weight = torch.randn(channels, channels, 1, 1) * 0.1
        self.bias = torch.randn(channels) * 0.05

    def __call__(self, x):
        graph = x.graph
        b, c, h, w = x.shape

        params = graph.register_weight_bias(
            self,
            "pointwise_conv2d_forward",
            1, 2, c
        )

        y = graph.add_node(
            "Pointwise",
            [x],
            params,
            (b, c, h, w),
            int_args={
                "batch_size": b,
                "input_channels": c,
                "input_height": h,
                "input_width": w,
                "output_channels": c,
                "use_bias": 1,
            }
        )

        return graph.add_unary("ReLU6", y)


class TestModel:
    def __init__(self):
        self.blocks = [PointwiseBlock(8) for _ in range(4)]

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def pytorch(self, x):
        for block in self.blocks:
            x = F.relu6(F.conv2d(x, block.weight, block.bias))
        return x


# ============================================================
# Prueba de rendimiento
# ============================================================

def main():
    torch.manual_seed(42)

    batch_size = 64
    repeats = 5

    x = torch.randn(batch_size, 8, 28, 28)
    model = TestModel()

    graph = FpgaGraph()

    print("\nConstruyendo DAG...")
    graph.build(model, x)

    print("\nCalentamiento...")
    graph.run(x)

    print("\n========== MEDICIONES ==========")

    times = []
    output = None

    for i in range(repeats):
        start = time.perf_counter()

        output = graph.run(x)

        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000
        times.append(elapsed_ms)

        print(f"Repetición {i+1}: {elapsed_ms:.3f} ms")

    mean = sum(times) / len(times)

    print("\n========== RESUMEN ==========")
    print(f"Batch: {batch_size}")
    print(f"Operaciones: {len(model.blocks) * 2}")
    print(f"Repeticiones: {repeats}")
    print(f"Latencia media: {mean:.3f} ms")
    print(f"Latencia mínima: {min(times):.3f} ms")
    print(f"Latencia máxima: {max(times):.3f} ms")

    # Comparación numérica con PyTorch
    with torch.no_grad():
        reference = model.pytorch(x)

    rmse = torch.sqrt(
        torch.mean((output - reference) ** 2)
    ).item()

    print(f"RMSE vs PyTorch: {rmse:.8e}")


if __name__ == "__main__":
    main()
