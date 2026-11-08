import math
import torch

from python.layers.BatchNorm2d import ManualBatchNorm2d


def calculate_rmse(reference, manual):
    squared_error_sum = 0.0
    total_values = reference.numel()

    reference_flat = reference.flatten().tolist()
    manual_flat = manual.flatten().tolist()

    for index in range(total_values):
        difference = (
            reference_flat[index]
            - manual_flat[index]
        )

        squared_error_sum += (
            difference * difference
        )

    return math.sqrt(
        squared_error_sum / total_values
    )


def calculate_max_error(reference, manual):
    max_error = 0.0

    reference_flat = reference.flatten().tolist()
    manual_flat = manual.flatten().tolist()

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
    batch_size,
    channels,
    height,
    width,
    eps=1e-5,
    affine=True,
):

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    torch.manual_seed(42)

    # -------------------------------------------------
    # Entrada
    # -------------------------------------------------

    input_tensor = torch.randn(
        batch_size,
        channels,
        height,
        width,
        dtype=torch.float32,
    )

    # -------------------------------------------------
    # BatchNorm PyTorch
    # -------------------------------------------------

    pytorch_layer = torch.nn.BatchNorm2d(
        num_features=channels,
        eps=eps,
        affine=affine,
        track_running_stats=True,
    )

    # Colocar parámetros conocidos
    if affine:

        with torch.no_grad():

            pytorch_layer.weight.copy_(
                torch.randn(
                    channels,
                    dtype=torch.float32,
                )
            )

            pytorch_layer.bias.copy_(
                torch.randn(
                    channels,
                    dtype=torch.float32,
                )
            )

    with torch.no_grad():

        pytorch_layer.running_mean.copy_(
            torch.randn(
                channels,
                dtype=torch.float32,
            )
        )

        # running_var debe ser >= 0
        pytorch_layer.running_var.copy_(
            torch.rand(
                channels,
                dtype=torch.float32,
            )
            + 0.1
        )

    # IMPORTANTE:
    # usar estadísticas acumuladas, no estadísticas del batch
    pytorch_layer.eval()

    with torch.no_grad():

        pytorch_output = pytorch_layer(
            input_tensor
        )

    # -------------------------------------------------
    # BatchNorm manual
    # -------------------------------------------------

    manual_layer = ManualBatchNorm2d(
        num_features=channels,
        eps=eps,
        affine=affine,
        bias=True,
    )

    if affine:

        manual_weight = (
            pytorch_layer.weight
            .detach()
            .cpu()
            .tolist()
        )

        manual_bias = (
            pytorch_layer.bias
            .detach()
            .cpu()
            .tolist()
        )

    else:

        manual_weight = None
        manual_bias = None

    manual_running_mean = (
        pytorch_layer.running_mean
        .detach()
        .cpu()
        .tolist()
    )

    manual_running_var = (
        pytorch_layer.running_var
        .detach()
        .cpu()
        .tolist()
    )

    manual_layer.load_parameters(
        weight=manual_weight,
        bias=manual_bias,
        running_mean=manual_running_mean,
        running_var=manual_running_var,
    )

    manual_input = (
        input_tensor
        .detach()
        .cpu()
        .tolist()
    )

    manual_output_list = manual_layer(
        manual_input
    )

    # Tensor solo para facilitar la comparación
    manual_output = torch.tensor(
        manual_output_list,
        dtype=torch.float32,
    )

    # -------------------------------------------------
    # Comparación
    # -------------------------------------------------

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

    print(f"eps:          {eps}")
    print(f"affine:       {affine}")

    print()
    print(f"RMSE:      {rmse:.10e}")
    print(f"Max error: {max_error:.10e}")

    tolerance = 1e-6

    if max_error <= tolerance:
        print("Resultado: PASS")
        return True

    print("Resultado: FAIL")
    return False


if __name__ == "__main__":

    results = []

    # -------------------------------------------------
    # Prueba 1: caso pequeño
    # -------------------------------------------------

    results.append(
        run_test(
            name="BatchNorm2D normal",
            batch_size=2,
            channels=4,
            height=8,
            width=8,
        )
    )

    # -------------------------------------------------
    # Prueba 2: estilo MobileNetV2
    # -------------------------------------------------

    results.append(
        run_test(
            name="BatchNorm2D MobileNetV2",
            batch_size=1,
            channels=32,
            height=8,
            width=8,
        )
    )

    # -------------------------------------------------
    # Prueba 3: affine=False
    # -------------------------------------------------

    results.append(
        run_test(
            name="BatchNorm2D sin affine",
            batch_size=2,
            channels=6,
            height=5,
            width=5,
            affine=False,
        )
    )

    # -------------------------------------------------
    # Prueba 4: eps diferente
    # -------------------------------------------------

    results.append(
        run_test(
            name="BatchNorm2D eps=1e-3",
            batch_size=1,
            channels=8,
            height=6,
            width=6,
            eps=1e-3,
        )
    )

    print()
    print("=" * 60)

    if all(results):
        print("TODAS LAS PRUEBAS: PASS")
    else:
        print("ALGUNA PRUEBA: FAIL")