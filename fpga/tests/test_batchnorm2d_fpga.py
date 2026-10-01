import torch
import torch.nn.functional as F

from fpga_layers.BatchNorm2d import FpgaBatchNorm2d


def run_test(
    affine: bool,
    bias: bool
) -> bool:

    torch.manual_seed(0)

    channels = 8

    input_tensor = torch.randn(
        1,
        channels,
        16,
        16,
        dtype=torch.float32
    )

    layer = FpgaBatchNorm2d(
        num_features=channels,
        eps=1e-5,
        affine=affine,
        bias=bias
    )

    # Valores no triviales para comprobar realmente
    # BatchNorm y no solamente identidad.
    with torch.no_grad():

        layer.running_mean.copy_(
            torch.randn(channels)
        )

        layer.running_var.copy_(
            torch.rand(channels) + 0.5
        )

        if affine:

            layer.weight.copy_(
                torch.randn(channels)
            )

            if layer.bias is not None:
                layer.bias.copy_(
                    torch.randn(channels)
                )

    reference_weight = (
        layer.weight
        if affine
        else None
    )

    reference_bias = (
        layer.bias
        if affine and bias
        else None
    )

    with torch.inference_mode():

        pytorch_output = F.batch_norm(
            input_tensor,
            layer.running_mean,
            layer.running_var,
            weight=reference_weight,
            bias=reference_bias,
            training=False,
            momentum=0.1,
            eps=layer.eps
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
    print(
        f"BATCHNORM2D - affine={affine}, bias={bias}"
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
        run_test(
            affine=True,
            bias=True
        ),
        run_test(
            affine=True,
            bias=False
        ),
        run_test(
            affine=False,
            bias=False
        ),
    ]

    print("\n" + "=" * 60)

    if all(results):
        print("RESULTADO GENERAL: PASS")
    else:
        print("RESULTADO GENERAL: FAIL")

    print("=" * 60)


if __name__ == "__main__":
    main()
