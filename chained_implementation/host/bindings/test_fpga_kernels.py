import numpy as np
import torch
import torch.nn.functional as F

import fpga_kernels


TOLERANCE = 1e-3

def check_result(name, fpga, reference):
    print(f"\n{name}")
    print(f"  Salida FPGA:  {fpga.shape}")
    print(f"  Salida torch: {reference.shape}")

    if fpga.shape != reference.shape:
        print("  Resultado:    FAIL (formas diferentes)")
        return False

    # Calculamos las métricas en float64 para reducir el
    # redondeo de la propia medición. Los kernels siguen en FP32.
    difference = (
        fpga.astype(np.float64)
        - reference.astype(np.float64)
    )

    rmse = float(np.sqrt(np.mean(difference ** 2)))
    max_error = float(np.max(np.abs(difference)))

    passed = (
        np.isfinite(rmse)
        and np.isfinite(max_error)
        and rmse <= TOLERANCE
        and max_error <= TOLERANCE
    )

    print(f"  RMSE:         {rmse:.10e}")
    print(f"  Error máximo: {max_error:.10e}")
    print(f"  Resultado:    {'PASS' if passed else 'FAIL'}")

    return bool(passed)

def test_conv2d():
    x = torch.randn(
        1, 1, 28, 28,
        dtype=torch.float32
    )

    weight = torch.randn(
        6, 1, 5, 5,
        dtype=torch.float32
    )

    bias = torch.randn(
        6,
        dtype=torch.float32
    )

    fpga = fpga_kernels.conv2d_forward(
        x.numpy(),
        weight.numpy(),
        bias.numpy(),
        1, 1,
        0, 0,
        True
    )

    reference = F.conv2d(
        x,
        weight,
        bias,
        stride=1,
        padding=0
    ).numpy()

    return check_result(
        "Conv2D",
        fpga,
        reference
    )


def test_tanh():
    x = torch.linspace(
        -3.0,
        3.0,
        1024,
        dtype=torch.float32
    )

    fpga = fpga_kernels.tanh_forward(
        x.numpy()
    )

    reference = torch.tanh(x).numpy()

    return check_result(
        "Tanh",
        fpga,
        reference
    )


def test_avgpool2d():
    x = torch.randn(
        1, 6, 24, 24,
        dtype=torch.float32
    )

    fpga = fpga_kernels.avgpool2d_forward(
        x.numpy(),
        2, 2,       # kernel
        2, 2,       # stride
        0, 0,       # padding
        False,      # count_include_pad
        False,      # use_divisor_override
        0           # divisor_override
    )

    reference = F.avg_pool2d(
        x,
        kernel_size=2,
        stride=2,
        padding=0,
        count_include_pad=False
    ).numpy()

    return check_result(
        "AvgPool2D",
        fpga,
        reference
    )


def test_linear():
    x = torch.randn(
        1, 400,
        dtype=torch.float32
    )

    weight = torch.randn(
        120, 400,
        dtype=torch.float32
    )

    bias = torch.randn(
        120,
        dtype=torch.float32
    )

    fpga = fpga_kernels.linear_forward(
        x.numpy(),
        weight.numpy(),
        bias.numpy(),
        True
    )

    reference = F.linear(
        x,
        weight,
        bias
    ).numpy()

    return check_result(
        "Linear",
        fpga,
        reference
    )

def test_relu6():
    # Incluye valores negativos, los límites 0 y 6,
    # valores dentro del intervalo y valores mayores que 6.
    x = torch.tensor(
        [-10.0, -1.0, 0.0, 0.5, 3.0, 5.9, 6.0, 6.1, 10.0],
        dtype=torch.float32
    ).reshape(1, 1, 3, 3)

    fpga = fpga_kernels.relu6_forward(x.numpy())
    reference = F.relu6(x).numpy()

    return check_result("ReLU6", fpga, reference)


def test_layeradd():
    x1 = torch.randn(1, 8, 8, 8, dtype=torch.float32)
    x2 = torch.randn(1, 8, 8, 8, dtype=torch.float32)

    results = []

    # Sin pasar alpha: comprueba el valor predeterminado 1.0.
    fpga = fpga_kernels.layeradd_forward(
        x1.numpy(),
        x2.numpy()
    )

    reference = torch.add(x1, x2).numpy()

    results.append(
        check_result("LayerAdd, alpha por defecto", fpga, reference)
    )

    # Comprueba que alpha escala la segunda entrada.
    alpha = 0.5

    fpga = fpga_kernels.layeradd_forward(
        x1.numpy(),
        x2.numpy(),
        alpha
    )

    reference = torch.add(x1, x2, alpha=alpha).numpy()

    results.append(
        check_result("LayerAdd, alpha=0.5", fpga, reference)
    )

    return all(results)


def test_depthwise():
    channels = 8

    x = torch.randn(1, channels, 16, 16, dtype=torch.float32)
    weight = torch.randn(channels, 1, 3, 3, dtype=torch.float32)
    bias = torch.randn(channels, dtype=torch.float32)

    results = []

    for stride in (1, 2):
        for use_bias in (False, True):
            fpga = fpga_kernels.depthwise_forward(
                x.numpy(),
                weight.numpy(),
                bias.numpy(),
                stride, stride,
                1, 1,
                use_bias
            )

            reference = F.conv2d(
                x,
                weight,
                bias if use_bias else None,
                stride=stride,
                padding=1,
                groups=channels
            ).numpy()

            results.append(
                check_result(
                    f"Depthwise, stride={stride}, use_bias={use_bias}",
                    fpga,
                    reference
                )
            )

    return all(results)


def test_pointwise():
    x = torch.randn(1, 8, 16, 16, dtype=torch.float32)
    weight = torch.randn(16, 8, 1, 1, dtype=torch.float32)
    bias = torch.randn(16, dtype=torch.float32)

    results = []

    for use_bias in (False, True):
        print(
            f"[Pointwise] Antes de FPGA, use_bias={use_bias}",
            flush=True
        )

        fpga = fpga_kernels.pointwise_forward(
            x.numpy(),
            weight.numpy(),
            bias.numpy(),
            use_bias
        )

        print("[Pointwise] FPGA terminó; iniciando PyTorch", flush=True)

        reference = F.conv2d(
            x,
            weight,
            bias if use_bias else None,
            stride=1,
            padding=0
        ).numpy()

        print("[Pointwise] PyTorch terminó", flush=True)

        results.append(
            check_result(
                f"Pointwise, use_bias={use_bias}",
                fpga,
                reference
            )
        )

    return all(results)


def test_batchnorm2d():
    channels = 8

    x = torch.randn(1, channels, 16, 16, dtype=torch.float32)
    weight = torch.randn(channels, dtype=torch.float32)
    bias = torch.randn(channels, dtype=torch.float32)
    running_mean = torch.randn(channels, dtype=torch.float32)

    # Varianzas positivas para una normalización válida.
    running_var = torch.rand(channels, dtype=torch.float32) + 0.5

    eps = 1e-5
    results = []

    for affine, use_bias in (
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ):
        fpga = fpga_kernels.batchnorm2d_forward(
            x.numpy(),
            weight.numpy(),
            bias.numpy(),
            running_mean.numpy(),
            running_var.numpy(),
            eps,
            affine,
            use_bias
        )

        reference = F.batch_norm(
            x,
            running_mean,
            running_var,
            weight=weight if affine else None,
            bias=bias if affine and use_bias else None,
            training=False,
            eps=eps
        ).numpy()

        results.append(
            check_result(
                f"BatchNorm2D, affine={affine}, use_bias={use_bias}",
                fpga,
                reference
            )
        )

    return all(results)

def main():
    torch.manual_seed(0)

    print("================================")
    print("Prueba conjunta de kernels FPGA")
    print("================================")

    results = []

    results.append(test_conv2d())
    results.append(test_tanh())
    results.append(test_avgpool2d())
    results.append(test_linear())

    results.append(test_relu6())
    results.append(test_layeradd())
    results.append(test_depthwise())
    results.append(test_pointwise())
    results.append(test_batchnorm2d())

    passed = sum(results)
    total = len(results)

    print("\n================================")
    print("Resumen")
    print("================================")
    print(f"Operaciones aprobadas: {passed}/{total}")

    if all(results):
        print("Resultado general: PASS")
        return

    print("Resultado general: FAIL")
    raise SystemExit(1)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
