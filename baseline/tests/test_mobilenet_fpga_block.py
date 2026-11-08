import torch
from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights,
)

from models.FpgaMobileNetV2 import FpgaConvBNReLU6


TOLERANCE = 1e-3


def main():

    print("======================================")
    print("MobileNetV2 - Primer bloque FPGA")
    print("Conv2D -> BatchNorm2D -> ReLU6")
    print("======================================")

    # --------------------------------------------------------
    # MobileNetV2 oficial
    # --------------------------------------------------------

    weights = MobileNet_V2_Weights.IMAGENET1K_V1

    pytorch_model = mobilenet_v2(
        weights=weights
    )

    pytorch_model.eval()

    # Primer bloque de torchvision:
    # Conv2D -> BatchNorm2D -> ReLU6

    pytorch_layer = pytorch_model.features[0]

    # --------------------------------------------------------
    # Bloque FPGA equivalente
    # --------------------------------------------------------

    fpga_layer = FpgaConvBNReLU6(
        in_channels=3,
        out_channels=32,
        kernel_size=3,
        stride=2,
        padding=1,
        groups=1,
    )

    # Copiar los pesos reales del bloque PyTorch
    fpga_layer.load_pytorch_layer(
        pytorch_layer
    )

    # --------------------------------------------------------
    # Entrada
    # --------------------------------------------------------

    torch.manual_seed(0)

    input_tensor = torch.randn(
        1,
        3,
        224,
        224,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Inferencia
    # --------------------------------------------------------

    with torch.inference_mode():

        output_torch = pytorch_layer(
            input_tensor
        )

        output_fpga = fpga_layer(
            input_tensor
        )

    # --------------------------------------------------------
    # Comparación
    # --------------------------------------------------------

    print(
        "Entrada:      ",
        tuple(input_tensor.shape)
    )

    print(
        "Salida torch: ",
        tuple(output_torch.shape)
    )

    print(
        "Salida FPGA:  ",
        tuple(output_fpga.shape)
    )

    if output_torch.shape != output_fpga.shape:

        print("RESULTADO: FAIL")
        print("Las dimensiones no coinciden.")

        raise SystemExit(1)

    difference = (
        output_fpga
        - output_torch
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
        rmse <= TOLERANCE
        and max_error <= TOLERANCE
    )

    print(
        f"RMSE:         {rmse:.10e}"
    )

    print(
        f"Error máximo: {max_error:.10e}"
    )

    print(
        "RESULTADO:   ",
        "PASS" if passed else "FAIL"
    )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
