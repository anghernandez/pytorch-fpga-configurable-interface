import torch

from fpga_layers.LayerAdd import FpgaLayerAdd


def run_test(alpha: float) -> bool:

    torch.manual_seed(0)

    input1 = torch.randn(
        1, 8, 16, 16,
        dtype=torch.float32
    )

    input2 = torch.randn(
        1, 8, 16, 16,
        dtype=torch.float32
    )

    layer = FpgaLayerAdd(
        alpha=alpha
    )

    with torch.inference_mode():

        pytorch_output = (
            input1
            + alpha * input2
        )

        fpga_output = layer(
            input1,
            input2
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
    print(
        f"LAYERADD - alpha={alpha}"
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

    results = [
        run_test(alpha=1.0),
        run_test(alpha=0.5),
    ]

    print("\n" + "=" * 60)

    if all(results):
        print("RESULTADO GENERAL: PASS")
    else:
        print("RESULTADO GENERAL: FAIL")

    print("=" * 60)


if __name__ == "__main__":
    main()
