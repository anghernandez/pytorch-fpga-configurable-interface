
import torch

from fpga_layers import FpgaConv2D


RMSE_TOLERANCE = 1e-3


def calculate_rmse(
    fpga_output: torch.Tensor,
    torch_output: torch.Tensor
) -> float:

    error = fpga_output - torch_output

    rmse = torch.sqrt(
        torch.mean(error ** 2)
    )

    return rmse.item()


def calculate_max_error(
    fpga_output: torch.Tensor,
    torch_output: torch.Tensor
) -> float:

    error = torch.abs(
        fpga_output - torch_output
    )

    return torch.max(error).item()


def run_test(
    name: str,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    input_height: int,
    input_width: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0
) -> None:

    torch.manual_seed(42)

    torch_layer = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True
    )

    fpga_layer = FpgaConv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True
    )

    with torch.no_grad():
        fpga_layer.weight.copy_(
            torch_layer.weight
        )

        fpga_layer.bias.copy_(
            torch_layer.bias
        )

    input_tensor = torch.randn(
        batch_size,
        in_channels,
        input_height,
        input_width,
        dtype=torch.float32
    )

    with torch.no_grad():
        torch_output = torch_layer(
            input_tensor
        )

        fpga_output = fpga_layer(
            input_tensor
        )

    rmse = calculate_rmse(
        fpga_output,
        torch_output
    )

    max_error = calculate_max_error(
        fpga_output,
        torch_output
    )

    passed = (
        fpga_output.shape == torch_output.shape
        and rmse <= RMSE_TOLERANCE
    )

    print(
        f"{name} | batch={batch_size} | "
        f"{in_channels} -> {out_channels}"
    )

    print(
        f"  Entrada:      {tuple(input_tensor.shape)}"
    )

    print(
        f"  Salida FPGA:  {tuple(fpga_output.shape)}"
    )

    print(
        f"  Salida torch: {tuple(torch_output.shape)}"
    )

    print(
        f"  RMSE:         {rmse:.10e}"
    )

    print(
        f"  Error máximo: {max_error:.10e}"
    )

    print(
        "  Resultado:    "
        + ("PASS" if passed else "FAIL")
    )

    print()


def main() -> None:

    print("==================")
    print("Pruebas FpgaConv2D")
    print("==================")
    print()

    for batch_size in (1, 16, 64):

        run_test(
            name="Conv1",
            batch_size=batch_size,
            in_channels=1,
            out_channels=6,
            input_height=28,
            input_width=28,
            kernel_size=5,
            stride=1,
            padding=0
        )

    for batch_size in (1, 16, 64):

        run_test(
            name="Conv2",
            batch_size=batch_size,
            in_channels=6,
            out_channels=16,
            input_height=14,
            input_width=14,
            kernel_size=5,
            stride=1,
            padding=0
        )


if __name__ == "__main__":
    main()

