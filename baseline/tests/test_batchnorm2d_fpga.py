import torch
import torch.nn.functional as F

from python_wrappers.BatchNorm2d import FpgaBatchNorm2d


TOLERANCE = 1e-3


def main() -> None:
    torch.manual_seed(0)

    batch_size = 128
    channels = 96
    height = 112
    width = 112

    print("=" * 60)
    print("BatchNorm2D aislada — affine=True, bias=True")
    print("Entrada:", (batch_size, channels, height, width))
    print("=" * 60, flush=True)

    input_tensor = torch.randn(
        batch_size,
        channels,
        height,
        width,
        dtype=torch.float32,
    )

    layer = FpgaBatchNorm2d(
        num_features=channels,
        eps=1e-5,
        affine=True,
        bias=True,
    )

    with torch.no_grad():
        layer.running_mean.copy_(torch.randn(channels))
        layer.running_var.copy_(torch.rand(channels) + 0.5)
        layer.weight.copy_(torch.randn(channels))
        layer.bias.copy_(torch.randn(channels))

    layer.eval()

    with torch.inference_mode():
        print("\nEjecutando BatchNorm FPGA...", flush=True)

        try:
            fpga_output = layer(input_tensor)
        except Exception:
            print(
                "\nFALLO durante la llamada BatchNorm FPGA.",
                flush=True,
            )
            raise

        print("BatchNorm FPGA completada.", flush=True)
        print("Forma FPGA:", tuple(fpga_output.shape), flush=True)

        if fpga_output.shape != input_tensor.shape:
            raise RuntimeError("La forma de salida no coincide con la entrada")

        print(
            "\nComparando con PyTorch, una imagen a la vez...",
            flush=True,
        )

        squared_error_sum = 0.0
        element_count = 0
        max_error = 0.0
        max_rmse_per_image = 0.0

        for index in range(batch_size):
            reference = F.batch_norm(
                input_tensor[index:index + 1],
                layer.running_mean,
                layer.running_var,
                weight=layer.weight,
                bias=layer.bias,
                training=False,
                momentum=0.1,
                eps=layer.eps,
            )

            output_slice = fpga_output[index:index + 1]

            if not (
                torch.isfinite(reference).all().item()
                and torch.isfinite(output_slice).all().item()
            ):
                raise RuntimeError(
                    f"NaN o infinito en la imagen {index}"
                )

            # FP64 para acumular el error numérico.
            difference = output_slice.double()
            difference.sub_(reference)

            image_max_error = difference.abs().max().item()
            difference.square_()

            image_squared_error = difference.sum().item()
            image_elements = difference.numel()
            image_rmse = (
                image_squared_error / image_elements
            ) ** 0.5

            squared_error_sum += image_squared_error
            element_count += image_elements
            max_error = max(max_error, image_max_error)
            max_rmse_per_image = max(
                max_rmse_per_image,
                image_rmse,
            )

            del reference, output_slice, difference

            if (index + 1) % 16 == 0:
                print(
                    f"Comparadas: {index + 1}/{batch_size}",
                    flush=True,
                )

    rmse = (squared_error_sum / element_count) ** 0.5
    passed = max_rmse_per_image <= TOLERANCE

    print("\n" + "=" * 60)
    print(f"RMSE global:           {rmse:.10e}")
    print(f"Mayor RMSE individual: {max_rmse_per_image:.10e}")
    print(f"Error máximo:          {max_error:.10e}")
    print("RESULTADO GENERAL:", "PASS" if passed else "FAIL")
    print("=" * 60)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()