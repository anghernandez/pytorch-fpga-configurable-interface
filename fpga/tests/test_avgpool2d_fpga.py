import torch
from torch import nn

from fpga_layers.AvgPool2d import FpgaAvgPool2D


TOLERANCE = 1e-3


def run_test(
    test_name: str,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    kernel_size,
    stride,
    padding=0,
    count_include_pad=True,
    divisor_override=None
) -> bool:

    print(
        f"\n{test_name} | "
        f"batch={batch_size} | "
        f"{channels}x{height}x{width}"
    )

    torch.manual_seed(0)

    input_tensor = torch.randn(
        batch_size,
        channels,
        height,
        width,
        dtype=torch.float32
    )

    fpga_pool = FpgaAvgPool2D(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override
    )

    torch_pool = nn.AvgPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override
    )

    with torch.no_grad():
        output_fpga = fpga_pool(
            input_tensor
        )

        output_torch = torch_pool(
            input_tensor
        )

    if output_fpga.shape != output_torch.shape:
        print(
            f"  Entrada:      "
            f"{tuple(input_tensor.shape)}"
        )

        print(
            f"  Salida FPGA:  "
            f"{tuple(output_fpga.shape)}"
        )

        print(
            f"  Salida torch: "
            f"{tuple(output_torch.shape)}"
        )

        print(
            "  Resultado:    FAIL "
            "(las dimensiones no coinciden)"
        )

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
        torch.abs(
            difference
        )
    ).item()

    passed = (
        rmse <= TOLERANCE
        and
        max_error <= TOLERANCE
    )

    print(
        f"  Entrada:      "
        f"{tuple(input_tensor.shape)}"
    )

    print(
        f"  Salida FPGA:  "
        f"{tuple(output_fpga.shape)}"
    )

    print(
        f"  Salida torch: "
        f"{tuple(output_torch.shape)}"
    )

    print(
        f"  RMSE:         "
        f"{rmse:.10e}"
    )

    print(
        f"  Error máximo: "
        f"{max_error:.10e}"
    )

    print(
        f"  Resultado:    "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return passed


def main():

    print("=====================")
    print("Pruebas FpgaAvgPool2D")
    print("=====================")

    all_passed = True

    # MobileNetV2 Global Average Pooling:
    #
    # Para una entrada de 224x224,
    # la salida final de features es:
    #
    # [N, 1280, 7, 7]
    #       ->
    # [N, 1280, 1, 1]
    #
    # Se reutiliza AvgPool2D haciendo que
    # el kernel cubra toda la dimensión espacial.

    all_passed &= run_test(
        test_name="MobileNetV2 GlobalAvgPool",
        batch_size=1,
        channels=1280,
        height=7,
        width=7,
        kernel_size=7,
        stride=7,
        padding=0,
        count_include_pad=True
    )


    # LeNet-5 Pool1:
    # [N, 6, 24, 24]
    #       ->
    # [N, 6, 12, 12]

    for batch_size in (1, 16, 64):
        all_passed &= run_test(
            test_name="Pool1",
            batch_size=batch_size,
            channels=6,
            height=24,
            width=24,
            kernel_size=2,
            stride=2
        )

    # LeNet-5 Pool2:
    # [N, 16, 8, 8]
    #       ->
    # [N, 16, 4, 4]

    for batch_size in (1, 16, 64):
        all_passed &= run_test(
            test_name="Pool2",
            batch_size=batch_size,
            channels=16,
            height=8,
            width=8,
            kernel_size=2,
            stride=2
        )

    print()

    if all_passed:
        print(
            "TODAS LAS PRUEBAS "
            "FPGA AVGPOOL2D PASARON"
        )
    else:
        print(
            "ALGUNA PRUEBA "
            "FPGA AVGPOOL2D FALLÓ"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
