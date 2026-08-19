# tests/test_relu.py

import torch
import torch.nn as nn

from layers.ReLU import ManualReLU


RMSE_LIMIT = 1e-3
ATOL = 1e-6
RTOL = 1e-5


def calculate_rmse(reference, prediction):
    """
    Calcula la raíz del error cuadrático medio.

    RMSE = sqrt(mean((reference - prediction)^2))
    """

    if reference.shape != prediction.shape:
        raise ValueError(
            "No se puede calcular el RMSE porque las formas "
            f"son diferentes: {reference.shape} y {prediction.shape}"
        )

    squared_error = (
        reference - prediction
    ) ** 2

    mean_squared_error = torch.mean(
        squared_error
    )

    rmse = torch.sqrt(
        mean_squared_error
    )

    return rmse.item()


def calculate_max_absolute_error(reference, prediction):
    """
    Calcula la diferencia absoluta máxima
    entre ambas salidas.
    """

    if reference.shape != prediction.shape:
        raise ValueError(
            "No se puede calcular el error máximo porque "
            f"las formas son diferentes: "
            f"{reference.shape} y {prediction.shape}"
        )

    absolute_error = torch.abs(
        reference - prediction
    )

    return torch.max(
        absolute_error
    ).item()


def test_manual_relu(shape):
    """
    Compara ManualReLU contra torch.nn.ReLU
    utilizando exactamente la misma entrada.

    También calcula:

        - Error absoluto máximo.
        - RMSE.
        - Coincidencia mediante torch.allclose.

    Parámetro:
        shape: forma de la entrada.
    """

    # ---------------------------------------------------------
    # 1. Reproducibilidad
    # ---------------------------------------------------------

    torch.manual_seed(5)

    # ---------------------------------------------------------
    # 2. Crear las implementaciones
    # ---------------------------------------------------------

    pytorch_relu = nn.ReLU()

    manual_relu = ManualReLU()

    # ---------------------------------------------------------
    # 3. Crear la entrada compartida
    # ---------------------------------------------------------

    x_torch = torch.randn(
        *shape,
        dtype=torch.float32,
    )

    x_manual = x_torch.tolist()

    # ---------------------------------------------------------
    # 4. Ejecutar PyTorch
    # ---------------------------------------------------------

    with torch.no_grad():
        y_pytorch = pytorch_relu(
            x_torch
        )

    # ---------------------------------------------------------
    # 5. Ejecutar ManualReLU
    # ---------------------------------------------------------

    y_manual_list = manual_relu(
        x_manual
    )

    y_manual = torch.tensor(
        y_manual_list,
        dtype=torch.float32,
    )

    # ---------------------------------------------------------
    # 6. Comparar formas
    # ---------------------------------------------------------

    same_shape = (
        y_pytorch.shape
        == y_manual.shape
    )

    # ---------------------------------------------------------
    # 7. Calcular métricas
    # ---------------------------------------------------------

    rmse = calculate_rmse(
        y_pytorch,
        y_manual,
    )

    max_error = calculate_max_absolute_error(
        y_pytorch,
        y_manual,
    )

    # ---------------------------------------------------------
    # 8. Comparar valores
    # ---------------------------------------------------------

    values_match = torch.allclose(
        y_pytorch,
        y_manual,
        atol=ATOL,
        rtol=RTOL,
    )

    rmse_is_valid = (
        rmse <= RMSE_LIMIT
    )

    # ---------------------------------------------------------
    # 9. Mostrar resultados
    # ---------------------------------------------------------

    print("=" * 60)
    print("VALIDACIÓN DE ManualReLU")
    print("=" * 60)

    print(f"Forma de entrada:       {shape}")
    print(f"Cantidad de elementos:  {x_torch.numel()}")
    print(f"Forma PyTorch:          {tuple(y_pytorch.shape)}")
    print(f"Forma ManualReLU:       {tuple(y_manual.shape)}")
    print(f"Misma forma:            {same_shape}")
    print(f"Error absoluto máximo:  {max_error:.10e}")
    print(f"RMSE:                   {rmse:.10e}")
    print(f"Límite RMSE:            {RMSE_LIMIT:.10e}")
    print(f"torch.allclose:          {values_match}")
    print(f"RMSE válido:            {rmse_is_valid}")

    if (
        same_shape
        and values_match
        and rmse_is_valid
    ):
        print(
            "Resultado:               APROBADO"
        )

    else:
        print(
            "Resultado:               NO APROBADO"
        )

    print("=" * 60)
    print()

    # ---------------------------------------------------------
    # 10. Verificación automática
    # ---------------------------------------------------------

    assert same_shape, (
        "La forma de salida de ManualReLU "
        "no coincide con PyTorch"
    )

    assert values_match, (
        "Los valores de ManualReLU "
        "no coinciden con torch.nn.ReLU"
    )

    assert rmse_is_valid, (
        f"El RMSE {rmse} supera el límite "
        f"permitido de {RMSE_LIMIT}"
    )

    return {
        "x": x_torch,
        "y_pytorch": y_pytorch,
        "y_manual": y_manual,
        "max_error": max_error,
        "rmse": rmse,
        "same_shape": same_shape,
        "values_match": values_match,
    }


def test_known_values():
    """
    Comprueba ManualReLU con valores conocidos.
    """

    manual_relu = ManualReLU()

    x_manual = [
        [-3.0, -1.0, 0.0],
        [1.0, 2.5, -4.5],
    ]

    expected_output = [
        [0.0, 0.0, 0.0],
        [1.0, 2.5, 0.0],
    ]

    output_manual = manual_relu(
        x_manual
    )

    assert output_manual == expected_output


def test_input_is_not_modified():
    """
    Comprueba que ManualReLU no modifique
    la lista de entrada original.
    """

    manual_relu = ManualReLU()

    x_manual = [
        [-2.0, 0.0, 3.0],
        [4.0, -5.0, 6.0],
    ]

    original_input = [
        row.copy()
        for row in x_manual
    ]

    manual_relu(
        x_manual
    )

    assert x_manual == original_input


def test_invalid_non_list_input():
    """
    Comprueba que ManualReLU rechace
    una entrada que no sea una lista.
    """

    manual_relu = ManualReLU()

    try:
        manual_relu(
            4.0
        )

    except TypeError:
        pass

    else:
        raise AssertionError(
            "ManualReLU debía lanzar TypeError "
            "para una entrada que no fuera una lista"
        )


def test_empty_input():
    """
    Comprueba que ManualReLU rechace
    una lista vacía.
    """

    manual_relu = ManualReLU()

    try:
        manual_relu(
            []
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "ManualReLU debía lanzar ValueError "
            "para una lista vacía"
        )


def test_nested_empty_list():
    """
    Comprueba que ManualReLU rechace
    listas vacías anidadas.
    """

    manual_relu = ManualReLU()

    try:
        manual_relu(
            [
                [1.0, 2.0],
                [],
            ]
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "ManualReLU debía lanzar ValueError "
            "para una lista vacía anidada"
        )


def test_invalid_element():
    """
    Comprueba que ManualReLU rechace
    elementos no numéricos.
    """

    manual_relu = ManualReLU()

    try:
        manual_relu(
            [
                [1.0, "dato inválido"],
            ]
        )

    except TypeError:
        pass

    else:
        raise AssertionError(
            "ManualReLU debía lanzar TypeError "
            "para elementos no numéricos"
        )


def main():
    """
    Ejecuta el banco completo de pruebas
    de ManualReLU.
    """

    test_shapes = [
        # Vector simple
        (10,),

        # Salida de una capa Linear
        (8, 16),

        # Entrada similar a un batch de MNIST
        (4, 1, 28, 28),

        # Activación intermedia similar a LeNet-5
        (4, 6, 24, 24),

        # Entrada de una imagen ImageNet
        (1, 3, 224, 224),

        # Activación interna similar a MobileNetV2
        (1, 32, 112, 112),
    ]

    results = []

    print()
    print("#" * 60)
    print("BANCO DE PRUEBAS DE ManualReLU")
    print("#" * 60)
    print()

    for shape in test_shapes:
        result = test_manual_relu(
            shape=shape,
        )

        results.append(
            result
        )

    # ---------------------------------------------------------
    # Pruebas adicionales
    # ---------------------------------------------------------

    test_known_values()

    test_input_is_not_modified()

    test_invalid_non_list_input()

    test_empty_input()

    test_nested_empty_list()

    test_invalid_element()

    # ---------------------------------------------------------
    # Resumen final
    # ---------------------------------------------------------

    print("#" * 60)
    print("RESUMEN GENERAL")
    print("#" * 60)

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"Prueba {index}: "
            f"shape={tuple(result['x'].shape)}, "
            f"RMSE={result['rmse']:.10e}, "
            f"error máximo={result['max_error']:.10e}"
        )

    print()
    print(
        "Pruebas de valores conocidos:     APROBADAS"
    )

    print(
        "Prueba de entrada no modificada:  APROBADA"
    )

    print(
        "Pruebas de validación de entrada: APROBADAS"
    )

    print()
    print(
        "TODAS LAS PRUEBAS DE ManualReLU "
        "FUERON SUPERADAS."
    )

    print("#" * 60)


if __name__ == "__main__":
    main()