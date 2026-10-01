import torch
from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights,
)

from models.FpgaMobileNetV2 import FpgaInvertedResidual


TOLERANCE = 1e-3


def run_test(
    test_name,
    pytorch_block,
    input_tensor,
    in_channels,
    out_channels,
    stride,
    expand_ratio,
):

    print()
    print("======================================")
    print(test_name)
    print("======================================")

    # --------------------------------------------------------
    # Bloque FPGA
    # --------------------------------------------------------

    fpga_block = FpgaInvertedResidual(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        expand_ratio=expand_ratio,
    )

    # Copiar pesos reales de torchvision
    fpga_block.load_pytorch_layer(
        pytorch_block
    )

    # --------------------------------------------------------
    # Inferencia
    # --------------------------------------------------------

    with torch.inference_mode():

        output_torch = pytorch_block(
            input_tensor
        )

        output_fpga = fpga_block(
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

    print(
        "Residual:     ",
        fpga_block.use_residual
    )

    if output_torch.shape != output_fpga.shape:

        print("RESULTADO: FAIL")
        print("Las dimensiones no coinciden.")

        return False

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

    return passed


def main():

    print("======================================")
    print("MobileNetV2 - InvertedResidual FPGA")
    print("======================================")

    # --------------------------------------------------------
    # MobileNetV2 oficial
    # --------------------------------------------------------

    weights = MobileNet_V2_Weights.IMAGENET1K_V1

    pytorch_model = mobilenet_v2(
        weights=weights
    )

    pytorch_model.eval()

    torch.manual_seed(0)

    all_passed = True

    # ========================================================
    # Caso 1
    #
    # features[1]
    # 32 -> 16
    # expand_ratio = 1
    # stride = 1
    # sin residual
    # ========================================================

    input_1 = torch.randn(
        1,
        32,
        112,
        112,
        dtype=torch.float32
    )

    all_passed &= run_test(
        test_name=(
            "Caso 1 - expand=1, stride=1, "
            "sin residual"
        ),
        pytorch_block=pytorch_model.features[1],
        input_tensor=input_1,
        in_channels=32,
        out_channels=16,
        stride=1,
        expand_ratio=1,
    )

    # ========================================================
    # Caso 2
    #
    # features[2]
    # 16 -> 24
    # expand_ratio = 6
    # stride = 2
    # sin residual
    # ========================================================

    input_2 = torch.randn(
        1,
        16,
        112,
        112,
        dtype=torch.float32
    )

    all_passed &= run_test(
        test_name=(
            "Caso 2 - expand=6, stride=2, "
            "sin residual"
        ),
        pytorch_block=pytorch_model.features[2],
        input_tensor=input_2,
        in_channels=16,
        out_channels=24,
        stride=2,
        expand_ratio=6,
    )

    # ========================================================
    # Caso 3
    #
    # features[3]
    # 24 -> 24
    # expand_ratio = 6
    # stride = 1
    # con residual
    # ========================================================

    input_3 = torch.randn(
        1,
        24,
        56,
        56,
        dtype=torch.float32
    )

    all_passed &= run_test(
        test_name=(
            "Caso 3 - expand=6, stride=1, "
            "con residual"
        ),
        pytorch_block=pytorch_model.features[3],
        input_tensor=input_3,
        in_channels=24,
        out_channels=24,
        stride=1,
        expand_ratio=6,
    )

    # --------------------------------------------------------
    # Resultado general
    # --------------------------------------------------------

    print()
    print("======================================")
    print(
        "RESULTADO GENERAL:",
        "PASS" if all_passed else "FAIL"
    )
    print("======================================")

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
