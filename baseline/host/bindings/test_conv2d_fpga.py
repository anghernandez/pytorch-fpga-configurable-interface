import numpy as np
import torch
import torch.nn.functional as F

import fpga_kernels


TOLERANCE = 1e-3


def run_conv_test(
    name,
    input_shape,
    output_channels,
    kernel_size,
    stride,
    padding,
    use_bias
):
    batch_size, input_channels, _, _ = input_shape

    input_size = int(np.prod(input_shape))

    weight_shape = (
        output_channels,
        input_channels,
        kernel_size[0],
        kernel_size[1]
    )

    weight_size = int(np.prod(weight_shape))

    input_tensor = (
        (
            torch.arange(
                input_size,
                dtype=torch.float32
            ) %  17
        ) -  8
    ).mul(0).add(
               (
            (
                torch.arange(
                    input_size,
                    dtype=torch.float32
                ) % 17
            ) - 8
        ).mul(0.05)
    ).reshape(input_shape)

    weight_tensor = (
        (
            torch.arange(
                weight_size,
                dtype=torch.float32
            ) %  11
        ) - 5
    ).mul(0.02).reshape(weight_shape)

    bias_tensor = (
        torch.arange(
            output_channels,
            dtype=torch.float32
        ) - 3
    ).mul(0.01)

    input_numpy = (
        input_tensor
        .contiguous()
        .numpy()
    )

    weight_numpy = (
        weight_tensor
        .contiguous()
        .numpy()
    )

    bias_numpy = (
        bias_tensor
        .contiguous()
        .numpy()
    )

    output_fpga = fpga_kernels.conv2d_forward(
        input_numpy,
        weight_numpy,
        bias_numpy,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        use_bias
    )

    reference_bias = (
        bias_tensor if use_bias else None
    )

    output_reference = F.conv2d(
        input_tensor,
        weight_tensor,
        reference_bias,
        stride=stride,
        padding=padding
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
    print(f"  Input:        {input_numpy.shape}")
    print(f"  Weight:       {weight_numpy.shape}")
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
        run_conv_test(
            name="Prueba 1: LeNet Conv1 con bias",
            input_shape=(1, 1, 28, 28),
            output_channels=6,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            use_bias=True
        )
    )

    results.append(
        run_conv_test(
            name="Prueba 2: LeNet Conv2 con bias",
            input_shape=(1, 6, 14, 14),
            output_channels=16,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
            use_bias=True
        )
    )

    results.append(
        run_conv_test(
            name="Prueba 3: LeNet Conv1 sin bias",
            input_shape=(1, 1, 28, 28),
            output_channels=6,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            use_bias=False
        )
    )

    results.append(
        expect_error(
            "Prueba 4: canales incompatibles",
            lambda: fpga_kernels.conv2d_forward(
                np.zeros(
                    (1, 1, 28, 28),
                    dtype=np.float32
                ),
                np.zeros(
                    (6, 2, 5, 5),
                    dtype=np.float32
                ),
                np.zeros(
                    (6,),
                    dtype=np.float32
                ),
                1, 1,
                2, 2,
                True
            )
        )
    )

    results.append(
        expect_error(
            "Prueba 5: bias incompatible",
            lambda: fpga_kernels.conv2d_forward(
                np.zeros(
                    (1, 1, 28, 28),
                    dtype=np.float32
                ),
                np.zeros(
                    (6, 1, 5, 5),
                    dtype=np.float32
                ),
                np.zeros(
                    (5,),
                    dtype=np.float32
                ),
                1, 1,
                2, 2,
                True
            )
        )
    )

    results.append(
        run_conv_test(
            name="Prueba 6: dimensiones superiores al depth HLS",
            input_shape=(64, 6, 28, 28),
            output_channels=32,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
            use_bias=True
        )
    )

    results.append(
        expect_error(
            "Prueba 7: entrada float64",
            lambda: fpga_kernels.conv2d_forward(
                np.zeros(
                    (1, 1, 28, 28),
                    dtype=np.float64
                ),
                np.zeros(
                    (6, 1, 5, 5),
                    dtype=np.float32
                ),
                np.zeros(
                    (6,),
                    dtype=np.float32
                ),
                1, 1,
                2, 2,
                True
            )
        )
    )

    passed_tests = sum(results)
    total_tests = len(results)

    print("\n================================")
    print("Resumen de Conv2D FPGA")
    print("================================")
    print(f"Pruebas aprobadas: {passed_tests}/{total_tests}")

    if all(results):
        print("Resultado general: PASS")
        return

    print("Resultado general: FAIL")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
