import torch
import torch.nn as nn

from python.layers.AvgPool2d import ManualAvgPool2d


TOLERANCE = 1e-5


def calculate_rmse(reference, prediction):

    if reference.shape != prediction.shape:
        raise ValueError(
            f"Formas diferentes: {reference.shape} y {prediction.shape}"
        )

    return torch.sqrt(
        torch.mean((reference - prediction) ** 2)
    ).item()


def run_avgpool2d_test(
    test_name,
    batch_size,
    input_channels,
    input_height,
    input_width,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):

    # ---------------------------------------------------------
    # Crear capa oficial de PyTorch
    # ---------------------------------------------------------

    pytorch_pool = nn.AvgPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    # ---------------------------------------------------------
    # Crear capa manual
    # ---------------------------------------------------------

    manual_pool = ManualAvgPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    # ---------------------------------------------------------
    # Crear entrada común
    # ---------------------------------------------------------

    x_torch = torch.randn(
        batch_size,
        input_channels,
        input_height,
        input_width,
        dtype=torch.float32,
    )

    x_manual = x_torch.detach().tolist()

    # ---------------------------------------------------------
    # Ejecutar PyTorch
    # ---------------------------------------------------------

    with torch.no_grad():
        output_pytorch = pytorch_pool(x_torch)

    # ---------------------------------------------------------
    # Ejecutar ManualAvgPool2d
    # ---------------------------------------------------------

    output_manual_list = manual_pool.forward(x_manual)

    output_manual = torch.tensor(
        output_manual_list,
        dtype=output_pytorch.dtype,
    )

    # ---------------------------------------------------------
    # Calcular métricas
    # ---------------------------------------------------------

    same_shape = (
        output_pytorch.shape
        == output_manual.shape
    )

    if same_shape:

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

        allclose = torch.allclose(
            output_pytorch,
            output_manual,
            atol=TOLERANCE,
            rtol=TOLERANCE,
        )

    else:

        rmse = float("inf")
        max_error = float("inf")
        allclose = False

    passed = (
        same_shape
        and allclose
        and rmse <= TOLERANCE
        and max_error <= TOLERANCE
    )

    return {
        "nombre": test_name,
        "entrada": tuple(x_torch.shape),
        "salida_pytorch": tuple(output_pytorch.shape),
        "salida_manual": tuple(output_manual.shape),
        "kernel": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "misma_forma": same_shape,
        "allclose": allclose,
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

    test_cases = [
        {
            "test_name": "Average pooling básico",
            "batch_size": 2,
            "input_channels": 3,
            "input_height": 8,
            "input_width": 8,
            "kernel_size": (2, 2),
            "stride": (2, 2),
            "padding": (0, 0),
            "ceil_mode": False,
            "count_include_pad": True,
            "divisor_override": None,
        },
        {
            "test_name": "Stride predeterminado",
            "batch_size": 2,
            "input_channels": 3,
            "input_height": 8,
            "input_width": 8,
            "kernel_size": (2, 2),
            "stride": None,
            "padding": (0, 0),
            "ceil_mode": False,
            "count_include_pad": True,
            "divisor_override": None,
        },
        {
            "test_name": "Padding contado en el promedio",
            "batch_size": 2,
            "input_channels": 3,
            "input_height": 7,
            "input_width": 8,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "ceil_mode": False,
            "count_include_pad": True,
            "divisor_override": None,
        },
        {
            "test_name": "Padding excluido del promedio",
            "batch_size": 2,
            "input_channels": 3,
            "input_height": 7,
            "input_width": 8,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "ceil_mode": False,
            "count_include_pad": False,
            "divisor_override": None,
        },
        {
            "test_name": "Kernel y stride rectangulares",
            "batch_size": 2,
            "input_channels": 3,
            "input_height": 7,
            "input_width": 8,
            "kernel_size": (2, 3),
            "stride": (2, 1),
            "padding": (0, 1),
            "ceil_mode": False,
            "count_include_pad": True,
            "divisor_override": None,
        },
        {
            "test_name": "Average pooling con ceil_mode",
            "batch_size": 2,
            "input_channels": 3,
            "input_height": 7,
            "input_width": 8,
            "kernel_size": (3, 2),
            "stride": (2, 3),
            "padding": (1, 0),
            "ceil_mode": True,
            "count_include_pad": True,
            "divisor_override": None,
        },
        {
            "test_name": "Average pooling con divisor_override",
            "batch_size": 2,
            "input_channels": 3,
            "input_height": 8,
            "input_width": 8,
            "kernel_size": (2, 2),
            "stride": (2, 2),
            "padding": (0, 0),
            "ceil_mode": False,
            "count_include_pad": True,
            "divisor_override": 5,
        },
    ]

    results = []

    for test_case in test_cases:
        results.append(
            run_avgpool2d_test(**test_case)
        )

    # ---------------------------------------------------------
    # Mostrar resultados individuales
    # ---------------------------------------------------------

    print("\n" + "=" * 120)
    print("RESULTADOS MANUALAVGPOOL2D VS PYTORCH")
    print("=" * 120)

    for result in results:

        print(f"\nPrueba: {result['nombre']}")
        print(f"Entrada:            {result['entrada']}")
        print(f"Salida PyTorch:     {result['salida_pytorch']}")
        print(f"Salida manual:      {result['salida_manual']}")
        print(f"Kernel:             {result['kernel']}")
        print(f"Stride:             {result['stride']}")
        print(f"Padding:            {result['padding']}")
        print(f"ceil_mode:          {result['ceil_mode']}")
        print(
            "count_include_pad:  "
            f"{result['count_include_pad']}"
        )
        print(
            "divisor_override:   "
            f"{result['divisor_override']}"
        )
        print(f"Misma forma:        {result['misma_forma']}")
        print(f"Allclose:           {result['allclose']}")
        print(f"RMSE:               {result['rmse']:.12f}")
        print(
            "Error máximo:       "
            f"{result['error_maximo']:.12f}"
        )
        print(f"Resultado:          {result['resultado']}")

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
            "ManualAvgPool2d superó todos los casos."
        )

    else:

        failed_tests = [
            result["nombre"]
            for result in results
            if result["resultado"] == "FALLIDA"
        ]

        raise AssertionError(
            "ManualAvgPool2d falló en: "
            + ", ".join(failed_tests)
        )


if __name__ == "__main__":
    main()