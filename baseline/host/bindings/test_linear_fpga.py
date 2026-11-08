import numpy as np
import torch
import torch.nn.functional as F

import fpga_kernels


TOLERANCE = 1e-3

BATCH_SIZE = 64
IN_FEATURES = 400
OUT_FEATURES = 120


def create_data():
    input_tensor = (
        (
            torch.arange(
                BATCH_SIZE * IN_FEATURES,
                dtype=torch.float32
            ) % 17
        ) - 8
    ).mul(0.05).reshape(
        BATCH_SIZE,
        IN_FEATURES
    )

    weight_tensor = (
        (
            torch.arange(
                OUT_FEATURES * IN_FEATURES,
                dtype=torch.float32
            ) % 13
        ) - 6
    ).mul(0.02).reshape(
        OUT_FEATURES,
        IN_FEATURES
    )

    bias_tensor = (
        (
            torch.arange(
                OUT_FEATURES,
                dtype=torch.float32
            ) % 7
        ) - 3
    ).mul(0.01)

    return input_tensor, weight_tensor, bias_tensor


def to_numpy(tensor):
    return (
        tensor
        .detach()
        .cpu()
        .contiguous()
        .numpy()
    )


def compare(name, output, reference):
    difference = output - reference

    rmse = float(
        np.sqrt(
            np.mean(difference ** 2)
        )
    )

    max_error = float(
        np.max(np.abs(difference))
    )

    passed = (
        rmse <= TOLERANCE
        and max_error <= TOLERANCE
    )

    print(f"\n{name}")
    print(f"  RMSE:         {rmse:.10e}")
    print(f"  Error máximo: {max_error:.10e}")
    print(f"  Resultado:    {'PASS' if passed else 'FAIL'}")

    return passed


def expect_error(name, operation):
    try:
        operation()

    except (ValueError, TypeError) as error:
        print(f"\n{name}")
        print(f"  Error detectado: {error}")
        print("  Resultado:       PASS")
        return True

    print(f"\n{name}")
    print("  No se detectó el error esperado")
    print("  Resultado:       FAIL")
    return False


def main():
    input_tensor, weight_tensor, bias_tensor = (
        create_data()
    )

    input_numpy = to_numpy(input_tensor)
    weight_numpy = to_numpy(weight_tensor)
    bias_numpy = to_numpy(bias_tensor)

    results = []


    # Prueba 1: NumPy con bias
    output = fpga_kernels.linear_forward(
        input_numpy,
        weight_numpy,
        bias_numpy,
        True
    )

    reference = (
        input_numpy @ weight_numpy.T
        + bias_numpy
    ).astype(np.float32)

    results.append(
        compare(
            "Prueba 1: NumPy con bias",
            output,
            reference
        )
    )


    # Prueba 2: PyTorch con bias
    output = fpga_kernels.linear_forward(
        input_numpy,
        weight_numpy,
        bias_numpy,
        True
    )

    reference = F.linear(
        input_tensor,
        weight_tensor,
        bias_tensor
    ).detach().numpy()

    results.append(
        compare(
            "Prueba 2: PyTorch con bias",
            output,
            reference
        )
    )


    # Prueba 3: PyTorch sin bias
    output = fpga_kernels.linear_forward(
        input_numpy,
        weight_numpy,
        bias_numpy,
        False
    )

    reference = F.linear(
        input_tensor,
        weight_tensor,
        None
    ).detach().numpy()

    results.append(
        compare(
            "Prueba 3: PyTorch sin bias",
            output,
            reference
        )
    )


    # Prueba 4: in_features incompatibles
    results.append(
        expect_error(
            "Prueba 4: in_features incompatibles",
            lambda: fpga_kernels.linear_forward(
                np.zeros((1, 400), dtype=np.float32),
                np.zeros((120, 399), dtype=np.float32),
                np.zeros((120,), dtype=np.float32),
                True
            )
        )
    )


    # Prueba 5: bias de tamaño incorrecto
    results.append(
        expect_error(
            "Prueba 5: bias incompatible",
            lambda: fpga_kernels.linear_forward(
                np.zeros((1, 400), dtype=np.float32),
                np.zeros((120, 400), dtype=np.float32),
                np.zeros((119,), dtype=np.float32),
                True
            )
        )
    )

    # Prueba 6: dimensiones de Linear de MobileNetV2
    mobilenet_input = (
        (
            torch.arange(
                64 * 1280,
                dtype=torch.float32
            ) % 17
        ) - 8
    ).mul(0.05).reshape(64, 1280)

    mobilenet_weight = (
        (
            torch.arange(
                1000 * 1280,
                dtype=torch.float32
            ) % 13
        ) - 6
    ).mul(0.02).reshape(1000, 1280)

    mobilenet_bias = (
        (
            torch.arange(
                1000,
                dtype=torch.float32
            ) % 7
        ) - 3
    ).mul(0.01)

    mobilenet_input_numpy = to_numpy(mobilenet_input)
    mobilenet_weight_numpy = to_numpy(mobilenet_weight)
    mobilenet_bias_numpy = to_numpy(mobilenet_bias)

    output = fpga_kernels.linear_forward(
        mobilenet_input_numpy,
        mobilenet_weight_numpy,
        mobilenet_bias_numpy,
        True
    )

    reference = F.linear(
        mobilenet_input,
        mobilenet_weight,
        mobilenet_bias
    ).detach().numpy()

    results.append(
        compare(
            "Prueba 6: Linear MobileNetV2 [64, 1280] -> [64, 1000]",
            output,
            reference
        )
    )

    # Prueba 7: tipo distinto de FP32
    results.append(
        expect_error(
            "Prueba 7: entrada float64",
            lambda: fpga_kernels.linear_forward(
                np.zeros((1, 400), dtype=np.float64),
                weight_numpy,
                bias_numpy,
                True
            )
        )
    )


    passed_tests = sum(results)
    total_tests = len(results)

    print("\n================================")
    print("Resumen de Linear FPGA")
    print("================================")
    print(f"Pruebas aprobadas: {passed_tests}/{total_tests}")

    if all(results):
        print("Resultado general: PASS")
        return

    print("Resultado general: FAIL")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
