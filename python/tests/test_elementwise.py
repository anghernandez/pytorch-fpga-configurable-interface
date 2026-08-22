import torch

from python.layers.ElementWise import ManualElementWise


def run_elementwise_test(
    test_name,
    input1_torch,
    input2_torch,
):
    pytorch_operations = {
        "add": torch.add,
        "sub": torch.sub,
        "mul": torch.mul,
        "div": torch.div,
    }

    input1_manual = input1_torch.tolist()
    input2_manual = input2_torch.tolist()

    results = []

    for operation_name, pytorch_function in pytorch_operations.items():

        manual_layer = ManualElementWise(
            operation=operation_name
        )

        manual_output = manual_layer.forward(
            input1_manual,
            input2_manual
        )

        manual_output_tensor = torch.tensor(
            manual_output,
            dtype=input1_torch.dtype
        )

        pytorch_output = pytorch_function(
            input1_torch,
            input2_torch
        )

        match = torch.allclose(
            manual_output_tensor,
            pytorch_output,
            atol=1e-9,
            rtol=1e-9
        )

        maximum_error = torch.max(
            torch.abs(
                manual_output_tensor - pytorch_output
            )
        ).item()

        results.append(
            {
                "prueba": test_name,
                "operacion": operation_name,
                "forma": tuple(input1_torch.shape),
                "coinciden": match,
                "error_maximo": maximum_error,
            }
        )

    return results


def main():

    # ============================================================
    # PRUEBA 1: VALORES DEFINIDOS
    # ============================================================

    input1_torch = torch.tensor(
        [
            [
                [10.0, -20.0, 30.0],
                [40.0, 50.0, -60.0]
            ],
            [
                [70.0, 80.0, 90.0],
                [-100.0, 110.0, 120.0]
            ]
        ],
        dtype=torch.float64
    )

    input2_torch = torch.tensor(
        [
            [
                [2.0, 4.0, -5.0],
                [8.0, -10.0, 12.0]
            ],
            [
                [14.0, -16.0, 18.0],
                [20.0, 22.0, -24.0]
            ]
        ],
        dtype=torch.float64
    )

    results = []

    results.extend(
        run_elementwise_test(
            test_name="Valores definidos",
            input1_torch=input1_torch,
            input2_torch=input2_torch,
        )
    )

    # ============================================================
    # PRUEBA 2: TENSORES ALEATORIOS 4D
    # ============================================================

    torch.manual_seed(123)

    shape = (2, 3, 4, 5)

    input1_random = torch.randn(
        *shape,
        dtype=torch.float64
    )

    input2_random = torch.randn(
        *shape,
        dtype=torch.float64
    )

    # Evitar divisores demasiado cercanos a cero
    input2_random = torch.where(
        torch.abs(input2_random) < 0.1,
        torch.full_like(input2_random, 0.1),
        input2_random
    )

    results.extend(
        run_elementwise_test(
            test_name="Aleatoria 4D",
            input1_torch=input1_random,
            input2_torch=input2_random,
        )
    )

    # ============================================================
    # MOSTRAR RESULTADOS
    # ============================================================

    print("\n" + "=" * 80)
    print("BANCO DE PRUEBAS: ManualElementWise vs PyTorch")
    print("=" * 80)

    for result in results:

        print(
            f"\nPrueba:        {result['prueba']}"
        )

        print(
            f"Operación:     {result['operacion'].upper()}"
        )

        print(
            f"Forma:         {result['forma']}"
        )

        print(
            f"¿Coinciden?:   {result['coinciden']}"
        )

        print(
            f"Error máximo:  {result['error_maximo']:.12f}"
        )

    # ============================================================
    # RESUMEN
    # ============================================================

    passed_tests = sum(
        result["coinciden"]
        for result in results
    )

    total_tests = len(results)

    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)

    print(
        f"Pruebas superadas: {passed_tests}/{total_tests}"
    )

    if passed_tests == total_tests:

        print(
            "ManualElementWise superó todas las pruebas."
        )

    else:

        print(
            "Hay pruebas que requieren revisión."
        )

        raise AssertionError(
            "ManualElementWise no coincide con PyTorch."
        )


if __name__ == "__main__":
    main()