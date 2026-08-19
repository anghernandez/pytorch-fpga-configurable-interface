import torch
import torch.nn as nn





from layers.Conv2D import ManualConv2d

def calculate_rmse(reference, prediction):

    if reference.shape != prediction.shape:
        raise ValueError(
            f"Formas diferentes: {reference.shape} y {prediction.shape}"
        )

    return torch.sqrt(
        torch.mean((reference - prediction) ** 2)
    ).item()

def run_conv2d_test(
    test_name,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    kernel_size,
    stride,
    padding,
    use_bias,
):

    # ---------------------------------------------------------
    # Crear capa oficial de PyTorch
    # ---------------------------------------------------------

    pytorch_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=1,
        groups=1,
        bias=use_bias,
        padding_mode="zeros",
    )

    # ---------------------------------------------------------
    # Crear capa manual
    # ---------------------------------------------------------

    manual_conv = ManualConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=1,
        groups=1,
        bias=use_bias,
        padding_mode="zeros",
    )

    # ---------------------------------------------------------
    # Copiar exactamente los mismos pesos
    # ---------------------------------------------------------

    manual_conv.weight = (
        pytorch_conv.weight
        .detach()
        .tolist()
    )

    # ---------------------------------------------------------
    # Copiar bias
    # ---------------------------------------------------------

    if pytorch_conv.bias is not None:
        manual_conv.bias = (
            pytorch_conv.bias
            .detach()
            .tolist()
        )
    else:
        manual_conv.bias = None

    # ---------------------------------------------------------
    # Crear entrada común
    # ---------------------------------------------------------

    x_torch = torch.randn(
        batch_size,
        in_channels,
        input_height,
        input_width,
        dtype=torch.float32,
    )

    x_manual = x_torch.detach().tolist()

    # ---------------------------------------------------------
    # Ejecutar PyTorch
    # ---------------------------------------------------------

    with torch.no_grad():
        output_pytorch = pytorch_conv(x_torch)

    # ---------------------------------------------------------
    # Ejecutar ManualConv2d
    # ---------------------------------------------------------

    output_manual_list = manual_conv.forward(x_manual)

    output_manual = torch.tensor(
        output_manual_list,
        dtype=output_pytorch.dtype,
    )

    # ---------------------------------------------------------
    # Calcular métricas
    # ---------------------------------------------------------

    rmse = calculate_rmse(
        output_pytorch,
        output_manual,
    )

    max_error = torch.max(
        torch.abs(
            output_pytorch
            - output_manual
        )
    ).item()

    same_shape = (
        output_pytorch.shape
        == output_manual.shape
    )

    allclose = torch.allclose(
        output_pytorch,
        output_manual,
        atol=1e-5,
        rtol=1e-5,
    )

    passed = (
        same_shape
        and allclose
        and rmse <= 1e-5
    )

    return {
        "nombre": test_name,
        "entrada": tuple(x_torch.shape),
        "salida": tuple(output_pytorch.shape),
        "kernel": kernel_size,
        "stride": stride,
        "padding": padding,
        "bias": use_bias,
        "rmse": rmse,
        "error_maximo": max_error,
        "resultado": "SUPERADA" if passed else "FALLIDA",
    }


def main():

    # ---------------------------------------------------------
    # Reproducibilidad
    # ---------------------------------------------------------

    torch.manual_seed(0)

    # ---------------------------------------------------------
    # Ejecutar diferentes configuraciones
    # ---------------------------------------------------------

    results = []

    results.append(
        run_conv2d_test(
            test_name="Convolución básica",
            batch_size=2,
            in_channels=2,
            out_channels=3,
            input_height=7,
            input_width=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(0, 0),
            use_bias=True,
        )
    )

    results.append(
        run_conv2d_test(
            test_name="Convolución con padding",
            batch_size=2,
            in_channels=2,
            out_channels=3,
            input_height=7,
            input_width=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            use_bias=True,
        )
    )

    results.append(
        run_conv2d_test(
            test_name="Convolución con stride",
            batch_size=2,
            in_channels=2,
            out_channels=3,
            input_height=7,
            input_width=8,
            kernel_size=(3, 3),
            stride=(2, 1),
            padding=(1, 1),
            use_bias=True,
        )
    )

    results.append(
        run_conv2d_test(
            test_name="Convolución sin bias",
            batch_size=2,
            in_channels=2,
            out_channels=3,
            input_height=7,
            input_width=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            use_bias=False,
        )
    )

    # ---------------------------------------------------------
    # Mostrar resultados individuales
    # ---------------------------------------------------------

    print("\n" + "=" * 120)
    print("RESULTADOS")
    print("=" * 120)

    for result in results:

        print(f"\nPrueba: {result['nombre']}")

        print(
            f"Entrada:      {result['entrada']}"
        )

        print(
            f"Salida:       {result['salida']}"
        )

        print(
            f"Kernel:       {result['kernel']}"
        )

        print(
            f"Stride:       {result['stride']}"
        )

        print(
            f"Padding:      {result['padding']}"
        )

        print(
            f"Bias:         {result['bias']}"
        )

        print(
            f"RMSE:         {result['rmse']:.12f}"
        )

        print(
            f"Error máximo: {result['error_maximo']:.12f}"
        )

        print(
            f"Resultado:    {result['resultado']}"
        )

    # ---------------------------------------------------------
    # Resumen
    # ---------------------------------------------------------

    passed_tests = sum(
        result["resultado"] == "SUPERADA"
        for result in results
    )

    total_tests = len(results)

    print("\n" + "=" * 120)
    print("RESUMEN")
    print("=" * 120)

    print(
        f"Pruebas superadas: "
        f"{passed_tests}/{total_tests}"
    )

    if passed_tests == total_tests:

        print(
            "ManualConv2d superó todos los casos."
        )

    else:

        print(
            "Hay casos que requieren revisión."
        )


if __name__ == "__main__":
    main()