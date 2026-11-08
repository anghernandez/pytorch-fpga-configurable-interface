import numpy as np
import torch
import torch.nn.functional as F

import fpga_kernels


TOLERANCE = 1e-3


def run_functional_test(
    name,
    shape,
    kernel_size,
    stride,
    padding,
    count_include_pad,
    divisor_override=None
):
    total_elements = int(np.prod(shape))

    input_tensor = (
        (
            torch.arange(
                total_elements,
                dtype=torch.float32
            ) % 31
        ) - 15
    ).div(10.0).reshape(shape)

    input_numpy = (
        input_tensor
        .contiguous()
        .numpy()
    )

    use_divisor_override = (
        divisor_override is not None
    )

    divisor_value = (
        divisor_override
        if use_divisor_override
        else 0
    )

    output_fpga = fpga_kernels.avgpool2d_forward(
        input_numpy,
        kernel_size[0],
        kernel_size[1],
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        count_include_pad,
        use_divisor_override,
        divisor_value
    )

    output_reference = F.avg_pool2d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=False,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override
    ).numpy()

    difference = (
        output_fpga - output_reference
    )

    rmse = float(
        np.sqrt(
            np.mean(difference ** 2)
        )
    )

    max_error = float(
        np.max(np.abs(difference))
    )

    passed = (
        output_fpga.shape == output_reference.shape
        and rmse <= TOLERANCE
        and max_error <= TOLERANCE
    )

    print(f"\n{name}")
    print(f"  Entrada:      {input_numpy.shape}")
    print(f"  Salida FPGA:  {output_fpga.shape}")
    print(f"  Salida torch: {output_reference.shape}")
    print(f"  RMSE:         {rmse:.10e}")
    print(f"  Error máximo: {max_error:.10e}")
    print(f"  Resultado:    {'PASS' if passed else 'FAIL'}")

    return passed


def expect_error(name, operation):
    try:
        operation()

    except (ValueError, TypeError) as error:
        message = str(error).splitlines()[0]

        print(f"\n{name}")
        print(f"  Error detectado: {message}")
        print("  Resultado:       PASS")
        return True

    print(f"\n{name}")
    print("  No se detectó el error esperado")
    print("  Resultado:       FAIL")
    return False


def main():
    results = []

    results.append(
        run_functional_test(
            name="Prueba 1: LeNet Pool 1",
            shape=(1, 6, 28, 28),
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
            count_include_pad=False
        )
    )

    results.append(
        run_functional_test(
            name="Prueba 2: LeNet Pool 2",
            shape=(1, 16, 10, 10),
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
            count_include_pad=False
        )
    )

    results.append(
        run_functional_test(
            name="Prueba 3: padding sin incluir",
            shape=(1, 1, 3, 3),
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(1, 1),
            count_include_pad=False
        )
    )

    results.append(
        run_functional_test(
            name="Prueba 4: padding incluido",
            shape=(1, 1, 3, 3),
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(1, 1),
            count_include_pad=True
        )
    )

    results.append(
        run_functional_test(
            name="Prueba 5: divisor_override",
            shape=(1, 1, 3, 3),
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(1, 1),
            count_include_pad=True,
            divisor_override=3
        )
    )

    results.append(
        expect_error(
            "Prueba 6: entrada sin formato NCHW",
            lambda: fpga_kernels.avgpool2d_forward(
                np.zeros((28, 28), dtype=np.float32),
                2, 2,
                2, 2,
                0, 0,
                False,
                False,
                0
            )
        )
    )

    results.append(
        run_functional_test(
            name="Prueba 7: dimensiones superiores al depth HLS",
            shape=(64, 16, 28, 28),
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
            count_include_pad=False
        )
    )

    results.append(
        expect_error(
            "Prueba 8: entrada float64",
            lambda: fpga_kernels.avgpool2d_forward(
                np.zeros((1, 6, 28, 28), dtype=np.float64),
                2, 2,
                2, 2,
                0, 0,
                False,
                False,
                0
            )
        )
    )

    passed_tests = sum(results)
    total_tests = len(results)

    print("\n================================")
    print("Resumen de AvgPool2D FPGA")
    print("================================")
    print(f"Pruebas aprobadas: {passed_tests}/{total_tests}")

    if all(results):
        print("Resultado general: PASS")
        return

    print("Resultado general: FAIL")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
