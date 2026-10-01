import torch
import torch.nn.functional as F

from fpga_layers.ReLU6 import FpgaReLU6


def main() -> None:

    torch.manual_seed(0)

    # Multiplicamos para garantizar una buena cantidad
    # de valores negativos y mayores que 6.
    input_tensor = (
        torch.randn(
            1,
            8,
            16,
            16,
            dtype=torch.float32
        )
        * 5.0
        + 2.0
    )

    layer = FpgaReLU6()

    with torch.inference_mode():

        pytorch_output = F.relu6(
            input_tensor
        )

        fpga_output = layer(
            input_tensor
        )

    difference = (
        pytorch_output
        - fpga_output
    )

    rmse = torch.sqrt(
        torch.mean(
            difference ** 2
        )
    ).item()

    max_error = torch.max(
        torch.abs(difference)
    ).item()

    passed = (
        pytorch_output.shape == fpga_output.shape
        and rmse <= 1e-3
    )

    print("\n" + "=" * 60)
    print("RELU6")
    print("=" * 60)

    print(
        "Forma PyTorch:",
        tuple(pytorch_output.shape)
    )

    print(
        "Forma FPGA:   ",
        tuple(fpga_output.shape)
    )

    print(
        "Entrada min:  ",
        input_tensor.min().item()
    )

    print(
        "Entrada max:  ",
        input_tensor.max().item()
    )

    print(
        "Salida min:   ",
        fpga_output.min().item()
    )

    print(
        "Salida max:   ",
        fpga_output.max().item()
    )

    print(
        f"RMSE:         {rmse:.10e}"
    )

    print(
        f"Error máximo: {max_error:.10e}"
    )

    print(
        "RESULTADO:",
        "PASS" if passed else "FAIL"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
