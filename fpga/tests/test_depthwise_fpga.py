import torch
import torch.nn.functional as F

from fpga_layers.DepthwiseConv2d import FpgaDepthwiseConv2d


def run_test(
    stride: int,
    bias: bool
) -> bool:

    torch.manual_seed(0)

    channels = 8
    height = 16
    width = 16

    input_tensor = torch.randn(
        1,
        channels,
        height,
        width,
        dtype=torch.float32
    )

    fpga_layer = FpgaDepthwiseConv2d(
        channels=channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias
    )

    with torch.inference_mode():

        pytorch_output = F.conv2d(
            input_tensor,
            fpga_layer.weight,
            bias=fpga_layer.bias,
            stride=stride,
            padding=1,
            groups=channels
        )

        fpga_output = fpga_layer(
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

    rmse_limit = 1e-3

    passed = (
        pytorch_output.shape == fpga_output.shape
        and rmse <= rmse_limit
    )

    print("\n" + "=" * 60)
    print(
        f"DEPTHWISE - stride={stride}, bias={bias}"
    )
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
        f"RMSE:         {rmse:.10e}"
    )

    print(
        f"Error máximo: {max_error:.10e}"
    )

    print(
        "RESULTADO:",
        "PASS" if passed else "FAIL"
    )

    return passed


def main() -> None:

    results = []

    results.append(
        run_test(
            stride=1,
            bias=False
        )
    )

    results.append(
        run_test(
            stride=1,
            bias=True
        )
    )

    results.append(
        run_test(
            stride=2,
            bias=False
        )
    )

    results.append(
        run_test(
            stride=2,
            bias=True
        )
    )

    print("\n" + "=" * 60)

    if all(results):
        print("RESULTADO GENERAL: PASS")
    else:
        print("RESULTADO GENERAL: FAIL")

    print("=" * 60)


if __name__ == "__main__":
    main()