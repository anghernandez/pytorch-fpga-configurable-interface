import math
import torch
import torch.nn.functional as F

from python.layers.GlobalAvgPool2d import ManualGlobalAvgPool2d


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
    input_tensor,
):

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    # ----------------------------------------
    # PyTorch
    # ----------------------------------------

    with torch.no_grad():

        pytorch_output = F.adaptive_avg_pool2d(
            input_tensor,
            output_size=(1, 1),
        )

    # ----------------------------------------
    # Manual
    # ----------------------------------------

    manual_layer = ManualGlobalAvgPool2d()

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

    tolerance = 1e-5

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
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    results.append(
        run_test(
            name="GlobalAvgPool2D valores controlados",
            input_tensor=controlled_input,
        )
    )

    # ----------------------------------------
    # Prueba 2: varios canales
    # ----------------------------------------

    torch.manual_seed(42)

    input_multichannel = torch.randn(
        2,
        4,
        5,
        5,
        dtype=torch.float32,
    )

    results.append(
        run_test(
            name="GlobalAvgPool2D varios canales",
            input_tensor=input_multichannel,
        )
    )

    # ----------------------------------------
    # Prueba 3: entrada rectangular
    # ----------------------------------------

    input_rectangular = torch.randn(
        1,
        8,
        6,
        10,
        dtype=torch.float32,
    )

    results.append(
        run_test(
            name="GlobalAvgPool2D entrada rectangular",
            input_tensor=input_rectangular,
        )
    )

    # ----------------------------------------
    # Prueba 4: estilo MobileNetV2
    # ----------------------------------------

    input_mobilenet = torch.randn(
        1,
        1280,
        7,
        7,
        dtype=torch.float32,
    )

    results.append(
        run_test(
            name="GlobalAvgPool2D estilo MobileNetV2",
            input_tensor=input_mobilenet,
        )
    )

    print()
    print("=" * 60)

    if all(results):
        print("TODAS LAS PRUEBAS: PASS")
    else:
        print("ALGUNA PRUEBA: FAIL")