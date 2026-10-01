import sys
from pathlib import Path

import torch
import torch.nn.functional as F


# ============================================================
# Configuración del proyecto
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = PROJECT_ROOT / "python"
CPP_DIR = PROJECT_ROOT / "cpp"

sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(CPP_DIR))


from wrappers import (
    CppPointwiseConv2d,
    CppDepthwiseConv2d,
    CppBatchNorm2d,
    CppReLU6,
    CppLayerAdd,
    CppGlobalAvgPool2d,
    CppLinear,
)


# ============================================================
# Configuración de validación
# ============================================================

RMSE_LIMIT = 1e-5
MAX_ERROR_LIMIT = 1e-5

torch.manual_seed(42)


# ============================================================
# Comparación
# ============================================================

def compare_outputs(
    name: str,
    torch_output: torch.Tensor,
    cpp_output: torch.Tensor
) -> bool:

    torch_output = torch_output.detach().cpu()
    cpp_output = cpp_output.detach().cpu()

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Forma PyTorch: {tuple(torch_output.shape)}"
    )

    print(
        f"Forma wrapper: {tuple(cpp_output.shape)}"
    )

    if torch_output.shape != cpp_output.shape:

        print("Resultado:     FAIL")
        print("Las formas no coinciden.")

        return False

    difference = (
        torch_output.float()
        - cpp_output.float()
    )

    rmse = torch.sqrt(
        torch.mean(
            difference ** 2
        )
    ).item()

    max_error = torch.max(
        torch.abs(difference)
    ).item()

    passed = (
        rmse <= RMSE_LIMIT
        and max_error <= MAX_ERROR_LIMIT
    )

    print(
        f"RMSE:          {rmse:.10e}"
    )

    print(
        f"Error máximo:  {max_error:.10e}"
    )

    print(
        f"Resultado:     {'PASS' if passed else 'FAIL'}"
    )

    return passed


# ============================================================
# PointwiseConv2D
# ============================================================

def test_pointwise() -> bool:

    batch_size = 2
    in_channels = 8
    out_channels = 16
    height = 7
    width = 7

    input_tensor = torch.randn(
        batch_size,
        in_channels,
        height,
        width
    )

    layer = CppPointwiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        bias=True
    )

    with torch.no_grad():

        torch_output = F.conv2d(
            input_tensor,
            layer.weight,
            layer.bias,
            stride=1,
            padding=0
        )

        cpp_output = layer(
            input_tensor
        )

    return compare_outputs(
        "CppPointwiseConv2d",
        torch_output,
        cpp_output
    )


# ============================================================
# DepthwiseConv2D
# ============================================================

def test_depthwise() -> bool:

    batch_size = 2
    channels = 8
    height = 9
    width = 9

    layer = CppDepthwiseConv2d(
        channels=channels,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=True
    )

    input_tensor = torch.randn(
        batch_size,
        channels,
        height,
        width
    )

    with torch.no_grad():

        torch_output = F.conv2d(
            input_tensor,
            layer.weight,
            layer.bias,
            stride=2,
            padding=1,
            groups=channels
        )

        cpp_output = layer(
            input_tensor
        )

    return compare_outputs(
        "CppDepthwiseConv2d",
        torch_output,
        cpp_output
    )


# ============================================================
# BatchNorm2D
# ============================================================

def test_batchnorm() -> bool:

    batch_size = 2
    channels = 8
    height = 7
    width = 7

    input_tensor = torch.randn(
        batch_size,
        channels,
        height,
        width
    )

    layer = CppBatchNorm2d(
        num_features=channels,
        eps=1e-5,
        affine=True
    )

    # Valores no triviales para probar realmente
    # weight, bias, mean y variance.

    with torch.no_grad():

        layer.weight.copy_(
            torch.randn(channels)
        )

        layer.bias.copy_(
            torch.randn(channels)
        )

        layer.running_mean.copy_(
            torch.randn(channels)
        )

        layer.running_var.copy_(
            torch.rand(channels) + 0.5
        )

        torch_output = F.batch_norm(
            input_tensor,
            layer.running_mean,
            layer.running_var,
            layer.weight,
            layer.bias,
            training=False,
            momentum=0.1,
            eps=layer.eps
        )

        cpp_output = layer(
            input_tensor
        )

    return compare_outputs(
        "CppBatchNorm2d",
        torch_output,
        cpp_output
    )


# ============================================================
# ReLU6
# ============================================================

def test_relu6() -> bool:

    input_tensor = (
        torch.randn(
            2,
            8,
            7,
            7
        )
        * 5.0
    )

    layer = CppReLU6()

    with torch.no_grad():

        torch_output = F.relu6(
            input_tensor
        )

        cpp_output = layer(
            input_tensor
        )

    return compare_outputs(
        "CppReLU6",
        torch_output,
        cpp_output
    )


# ============================================================
# LayerAdd
# ============================================================

def test_layer_add() -> bool:

    input1 = torch.randn(
        2,
        8,
        7,
        7
    )

    input2 = torch.randn(
        2,
        8,
        7,
        7
    )

    alpha = 1.0

    layer = CppLayerAdd(
        alpha=alpha
    )

    with torch.no_grad():

        torch_output = (
            input1
            + alpha * input2
        )

        cpp_output = layer(
            input1,
            input2
        )

    return compare_outputs(
        "CppLayerAdd",
        torch_output,
        cpp_output
    )


# ============================================================
# GlobalAvgPool2D
# ============================================================

def test_global_avgpool() -> bool:

    input_tensor = torch.randn(
        2,
        8,
        7,
        7
    )

    layer = CppGlobalAvgPool2d()

    with torch.no_grad():

        torch_output = (
            F.adaptive_avg_pool2d(
                input_tensor,
                (1, 1)
            )
        )

        cpp_output = layer(
            input_tensor
        )

    return compare_outputs(
        "CppGlobalAvgPool2d",
        torch_output,
        cpp_output
    )


# ============================================================
# Linear
# ============================================================

def test_linear() -> bool:

    batch_size = 4
    in_features = 32
    out_features = 10

    input_tensor = torch.randn(
        batch_size,
        in_features
    )

    layer = CppLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True
    )

    with torch.no_grad():

        torch_output = F.linear(
            input_tensor,
            layer.weight,
            layer.bias
        )

        cpp_output = layer(
            input_tensor
        )

    return compare_outputs(
        "CppLinear",
        torch_output,
        cpp_output
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("\n" + "=" * 60)
    print("VALIDACIÓN PYTORCH vs WRAPPERS C++")
    print("=" * 60)

    results = []

    results.append(
        test_pointwise()
    )

    results.append(
        test_depthwise()
    )

    results.append(
        test_batchnorm()
    )

    results.append(
        test_relu6()
    )

    results.append(
        test_layer_add()
    )

    results.append(
        test_global_avgpool()
    )

    results.append(
        test_linear()
    )

    passed_tests = sum(
        results
    )

    total_tests = len(
        results
    )

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)

    print(
        f"Tests superados: "
        f"{passed_tests}/{total_tests}"
    )

    if passed_tests == total_tests:

        print(
            "RESULTADO GLOBAL: PASS"
        )

    else:

        print(
            "RESULTADO GLOBAL: FAIL"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
