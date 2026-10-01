import torch
import torch.nn.functional as F

from fpga_layers.PointwiseConv2d import FpgaPointwiseConv2d

def main() -> None:

    torch.manual_seed(0)

    # ============================================================
    # Configuración
    # ============================================================

    batch_size = 1
    in_channels = 8
    out_channels = 16
    height = 16
    width = 16

    # ============================================================
    # Entrada
    # ============================================================

    input_tensor = torch.randn(
        batch_size,
        in_channels,
        height,
        width,
        dtype=torch.float32
    )

    # ============================================================
    # Capa FPGA
    # ============================================================

    fpga_layer = FpgaPointwiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        bias=False
    )

    # ============================================================
    # Referencia PyTorch
    # ============================================================

    with torch.inference_mode():

        pytorch_output = F.conv2d(
            input_tensor,
            fpga_layer.weight,
            bias=None,
            stride=1,
            padding=0
        )

        fpga_output = fpga_layer(
            input_tensor
        )

    # ============================================================
    # Comparación
    # ============================================================

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

    # ============================================================
    # Resultados
    # ============================================================

    print("\n" + "=" * 60)
    print("POINTWISE CONV2D - PYTORCH vs FPGA")
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
        f"\nRMSE:         {rmse:.10e}"
    )

    print(
        f"Error máximo: {max_error:.10e}"
    )

    rmse_limit = 1e-3

    if (
        pytorch_output.shape == fpga_output.shape
        and rmse <= rmse_limit
    ):
        print("\nRESULTADO: PASS")
    else:
        print("\nRESULTADO: FAIL")


if __name__ == "__main__":
    main()