import torch
from torch import nn

from fpga_layers import FpgaLinear


def test_linear(
    name: str,
    batch_size: int,
    in_features: int,
    out_features: int,
    tolerance: float = 1e-3
) -> bool:

    # Entrada de prueba
    input_tensor = torch.randn(
        batch_size,
        in_features,
        dtype=torch.float32
    )

    # Referencia PyTorch
    torch_linear = nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=True
    )

    # Implementación FPGA
    fpga_linear = FpgaLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True
    )

    # Copiar los mismos pesos y bias
    with torch.no_grad():
        fpga_linear.weight.copy_(
            torch_linear.weight
        )

        fpga_linear.bias.copy_(
            torch_linear.bias
        )

    # Ejecutar PyTorch
    with torch.inference_mode():
        output_torch = torch_linear(
            input_tensor
        )

    # Ejecutar FPGA
    with torch.inference_mode():
        output_fpga = fpga_linear(
            input_tensor
        )

    # Calcular errores
    difference = output_fpga - output_torch

    rmse = torch.sqrt(
        torch.mean(
            difference ** 2
        )
    ).item()

    max_error = torch.max(
        torch.abs(difference)
    ).item()

    passed = rmse <= tolerance

    # Resultados
    print(
        f"{name} | "
        f"batch={batch_size} | "
        f"{in_features} -> {out_features}"
    )

    print(
        f"  Entrada:      {tuple(input_tensor.shape)}"
    )

    print(
        f"  Salida FPGA:  {tuple(output_fpga.shape)}"
    )

    print(
        f"  Salida torch: {tuple(output_torch.shape)}"
    )

    print(
        f"  RMSE:         {rmse:.10e}"
    )

    print(
        f"  Error máximo: {max_error:.10e}"
    )

    if passed:
        print("  Resultado:    PASS")
    else:
        print("  Resultado:    FAIL")

    print()

    return passed


def main() -> None:

    # Para reproducibilidad
    torch.manual_seed(42)

    # Configuraciones reales de LeNet-5
    linear_layers = [
        ("FC1", 400, 120),
        ("FC2", 120, 84),
        ("FC3", 84, 10),
    ]

    # Batch sizes que queremos validar
    batch_sizes = [
        1,
        16,
        64,
    ]

    total_tests = 0
    passed_tests = 0

    print("Pruebas FpgaLinear")
    print("==================")
    print()

    for name, in_features, out_features in linear_layers:

        for batch_size in batch_sizes:

            total_tests += 1

            passed = test_linear(
                name=name,
                batch_size=batch_size,
                in_features=in_features,
                out_features=out_features
            )

            if passed:
                passed_tests += 1

    print("==================")
    print("Resumen")
    print("==================")

    print(
        f"Pruebas superadas: "
        f"{passed_tests}/{total_tests}"
    )

    if passed_tests == total_tests:
        print(
            "Resultado global: PASS"
        )
    else:
        print(
            "Resultado global: FAIL"
        )


if __name__ == "__main__":
    main()