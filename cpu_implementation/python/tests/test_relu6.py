import math
import torch

from python.layers.ReLU6 import ManualReLU6


def calculate_rmse(reference, manual):

    reference_flat = reference.flatten().tolist()
    manual_flat = manual.flatten().tolist()

    squared_error_sum = 0.0

    for index in range(len(reference_flat)):

        difference = (
            reference_flat[index]
            - manual_flat[index]
        )

        squared_error_sum += (
            difference * difference
        )

    return math.sqrt(
        squared_error_sum / len(reference_flat)
    )


def calculate_max_error(reference, manual):

    reference_flat = reference.flatten().tolist()
    manual_flat = manual.flatten().tolist()

    max_error = 0.0

    for index in range(len(reference_flat)):

        error = abs(
            reference_flat[index]
            - manual_flat[index]
        )

        if error > max_error:
            max_error = error

    return max_error


def run_test(
    name,
    input_tensor
):

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    # ----------------------------------------
    # PyTorch
    # ----------------------------------------

    pytorch_layer = torch.nn.ReLU6()

    with torch.no_grad():

        pytorch_output = pytorch_layer(
            input_tensor
        )

    # ----------------------------------------
    # Manual
    # ----------------------------------------

    manual_layer = ManualReLU6()

    manual_input = (
        input_tensor
        .detach()
        .cpu()
        .tolist()
    )

    manual_output_list = manual_layer(
        manual_input
    )

    manual_output = torch.tensor(
        manual_output_list,
        dtype=torch.float32,
    )

    # ----------------------------------------
    # Comparación
    # ----------------------------------------

    rmse = calculate_rmse(
        pytorch_output,
        manual_output,
    )

    max_error = calculate_max_error(
        pytorch_output,
        manual_output,
    )

    print("Input shape: ", tuple(input_tensor.shape))
    print("Output shape:", tuple(pytorch_output.shape))

    print()
    print(f"RMSE:      {rmse:.10e}")
    print(f"Max error: {max_error:.10e}")

    tolerance = 1e-7

    if max_error <= tolerance:
        print("Resultado: PASS")
        return True

    print("Resultado: FAIL")
    return False


if __name__ == "__main__":

    results = []

    # ----------------------------------------
    # Prueba 1: valores controlados
    # ----------------------------------------

    controlled_input = torch.tensor(
        [
            -10.0,
            -1.0,
            0.0,
            1.5,
            6.0,
            8.0,
            100.0,
        ],
        dtype=torch.float32,
    )

    results.append(
        run_test(
            name="ReLU6 valores controlados",
            input_tensor=controlled_input,
        )
    )

    # ----------------------------------------
    # Prueba 2: tensor 2D
    # ----------------------------------------

    torch.manual_seed(42)

    input_2d = (
        torch.randn(
            4,
            8,
            dtype=torch.float32,
        )
        * 5.0
    )

    results.append(
        run_test(
            name="ReLU6 tensor 2D",
            input_tensor=input_2d,
        )
    )

    # ----------------------------------------
    # Prueba 3: formato tipo CNN
    # ----------------------------------------

    input_4d = (
        torch.randn(
            2,
            32,
            8,
            8,
            dtype=torch.float32,
        )
        * 5.0
    )

    results.append(
        run_test(
            name="ReLU6 tensor NCHW",
            input_tensor=input_4d,
        )
    )

    print()
    print("=" * 60)

    if all(results):
        print("TODAS LAS PRUEBAS: PASS")
    else:
        print("ALGUNA PRUEBA: FAIL")