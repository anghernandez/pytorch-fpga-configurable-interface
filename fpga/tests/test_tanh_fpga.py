import torch

from fpga_layers import FpgaTanh


def main():
    layer = FpgaTanh()

    input_tensor = torch.tensor(
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        dtype=torch.float32
    )

    output_fpga = layer(input_tensor)

    output_pytorch = torch.tanh(
        input_tensor
    )

    difference = (
        output_fpga - output_pytorch
    )

    rmse = torch.sqrt(
        torch.mean(difference ** 2)
    ).item()

    max_error = torch.max(
        torch.abs(difference)
    ).item()

    print("Entrada:")
    print(input_tensor)

    print("\nSalida FPGA:")
    print(output_fpga)

    print("\nReferencia PyTorch:")
    print(output_pytorch)

    print(f"\nRMSE:      {rmse:.10e}")
    print(f"Max error: {max_error:.10e}")

    if rmse <= 1e-3:
        print("Resultado: PASS")
    else:
        print("Resultado: FAIL")


if __name__ == "__main__":
    main()
