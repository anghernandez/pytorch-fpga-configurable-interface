import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Importar cpp_kernels
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CPP_DIR = PROJECT_ROOT / "cpp"

sys.path.insert(0, str(CPP_DIR))

import cpp_kernels


# ============================================================
# Configuración
# ============================================================

RMSE_LIMIT = 1e-5
MAX_ERROR_LIMIT = 1e-5

torch.manual_seed(42)


# ============================================================
# Utilidades
# ============================================================

def compare_outputs(name, torch_output, cpp_output):
    torch_np = (
        torch_output
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    cpp_np = np.asarray(
        cpp_output,
        dtype=np.float32
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Forma PyTorch: {torch_np.shape}")
    print(f"Forma C++:     {cpp_np.shape}")

    if torch_np.shape != cpp_np.shape:
        print("Resultado:     FAIL")
        print("Motivo: formas diferentes")
        return False

    difference = torch_np - cpp_np

    rmse = np.sqrt(
        np.mean(difference ** 2)
    )

    max_error = np.max(
        np.abs(difference)
    )

    passed = (
        rmse <= RMSE_LIMIT
        and max_error <= MAX_ERROR_LIMIT
    )

    print(f"RMSE:          {rmse:.10e}")
    print(f"Error máximo:  {max_error:.10e}")
    print(
        f"Resultado:     "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return passed

def test_pointwise_conv2d():
    x = torch.randn(
        2, 8, 7, 7,
        dtype=torch.float32
    )

    weight = torch.randn(
        16, 8, 1, 1,
        dtype=torch.float32
    )

    bias = torch.randn(
        16,
        dtype=torch.float32
    )

    torch_output = F.conv2d(
        x,
        weight,
        bias=bias,
        stride=1,
        padding=0
    )

    cpp_output = cpp_kernels.pointwise_conv2d_forward(
        x.numpy(),
        weight.numpy(),
        bias.numpy()
    )

    return compare_outputs(
        "PointwiseConv2D",
        torch_output,
        cpp_output
    )

def test_depthwise_conv2d():
    x = torch.randn(
        2, 8, 9, 9,
        dtype=torch.float32
    )

    weight = torch.randn(
        8, 1, 3, 3,
        dtype=torch.float32
    )

    bias = torch.randn(
        8,
        dtype=torch.float32
    )

    stride = 2
    padding = 1

    torch_output = F.conv2d(
        x,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        groups=8
    )

    output_height = torch_output.shape[2]
    output_width = torch_output.shape[3]

    cpp_output = cpp_kernels.depthwise_conv2d_forward(
        x.numpy(),
        weight.numpy(),
        bias.numpy(),
        output_height,
        output_width,
        stride,
        stride,
        padding,
        padding
    )

    return compare_outputs(
        "DepthwiseConv2D",
        torch_output,
        cpp_output
    )
def test_batchnorm2d():
    x = torch.randn(
        2, 8, 7, 7,
        dtype=torch.float32
    )

    weight = torch.randn(
        8,
        dtype=torch.float32
    )

    bias = torch.randn(
        8,
        dtype=torch.float32
    )

    running_mean = torch.randn(
        8,
        dtype=torch.float32
    )

    running_var = (
        torch.rand(
            8,
            dtype=torch.float32
        ) + 0.5
    )

    eps = 1e-5

    torch_output = F.batch_norm(
        x,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        training=False,
        eps=eps
    )

    cpp_output = cpp_kernels.batchnorm2d_forward(
        x.numpy(),
        weight.numpy(),
        bias.numpy(),
        running_mean.numpy(),
        running_var.numpy(),
        eps,
        True
    )

    return compare_outputs(
        "BatchNorm2D",
        torch_output,
        cpp_output
    )
def test_relu6():
    x = torch.randn(
        2, 8, 7, 7,
        dtype=torch.float32
    ) * 5.0

    torch_output = F.relu6(x)

    cpp_output = cpp_kernels.relu6_forward(
        x.numpy()
    )

    return compare_outputs(
        "ReLU6",
        torch_output,
        cpp_output
    )

def test_layer_add():
    input1 = torch.randn(
        2, 8, 7, 7,
        dtype=torch.float32
    )

    input2 = torch.randn(
        2, 8, 7, 7,
        dtype=torch.float32
    )

    alpha = 1.0

    torch_output = (
        input1 + alpha * input2
    )

    cpp_output = cpp_kernels.layer_add_forward(
        input1.numpy(),
        input2.numpy(),
        alpha
    )

    return compare_outputs(
        "LayerAdd",
        torch_output,
        cpp_output
    )

def test_global_avgpool2d():
    x = torch.randn(
        2, 8, 7, 7,
        dtype=torch.float32
    )

    torch_output = F.adaptive_avg_pool2d(
        x,
        output_size=(1, 1)
    )

    cpp_output = (
        cpp_kernels.global_avgpool2d_forward(
            x.numpy()
        )
    )

    return compare_outputs(
        "GlobalAvgPool2D",
        torch_output,
        cpp_output
    )

def test_linear():
    x = torch.randn(
        4, 32,
        dtype=torch.float32
    )

    weight = torch.randn(
        10, 32,
        dtype=torch.float32
    )

    bias = torch.randn(
        10,
        dtype=torch.float32
    )

    torch_output = F.linear(
        x,
        weight,
        bias
    )

    cpp_output = cpp_kernels.linear_forward(
        x.numpy(),
        weight.numpy(),
        bias.numpy()
    )

    return compare_outputs(
        "Linear",
        torch_output,
        cpp_output
    )

def main():
    print("\n")
    print("=" * 60)
    print("VALIDACIÓN PYTORCH vs KERNELS C++")
    print("=" * 60)

    results = []

    results.append(
        test_pointwise_conv2d()
    )

    results.append(
        test_depthwise_conv2d()
    )

    results.append(
        test_batchnorm2d()
    )

    results.append(
        test_relu6()
    )

    results.append(
        test_layer_add()
    )

    results.append(
        test_global_avgpool2d()
    )

    results.append(
        test_linear()
    )

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests superados: {passed}/{total}")

    if all(results):
        print("RESULTADO GLOBAL: PASS")
    else:
        print("RESULTADO GLOBAL: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()


