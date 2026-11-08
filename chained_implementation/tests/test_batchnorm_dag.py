import torch
import torch.nn.functional as F

from fpga_graph import FpgaGraph
from BatchNorm2d import FpgaBatchNorm2d


def main():
    torch.manual_seed(7)

    # Configuración de la prueba
    batch_size = 1
    channels = 2
    height = 4
    width = 4

    x = torch.randn(batch_size, channels, height, width)

    # Capa FPGA
    model = FpgaBatchNorm2d(channels).eval()

    # Construcción del DAG
    graph = FpgaGraph().build(model, x)

    # Inferencia FPGA
    with torch.inference_mode():
        y_fpga = graph.run(x)

    # Referencia PyTorch
    with torch.inference_mode():
        y_torch = F.batch_norm(
            x,
            model.running_mean,
            model.running_var,
            model.weight,
            model.bias,
            training=False,
            eps=model.eps
        )

    # Comparación
    difference = y_fpga - y_torch

    rmse = difference.square().mean().sqrt().item()
    max_error = difference.abs().max().item()

    print("\n========== BatchNorm2D ==========")
    print("Entrada:", tuple(x.shape))
    print("Salida FPGA:", tuple(y_fpga.shape))
    print("Salida PyTorch:", tuple(y_torch.shape))
    print(f"RMSE: {rmse:.10e}")
    print(f"Error máximo: {max_error:.10e}")
    print("FPGA:", y_fpga.flatten()[:8])
    print("PyTorch:", y_torch.flatten()[:8])
    print("PASS" if rmse < 1e-4 else "FAIL")


if __name__ == "__main__":
    main()
