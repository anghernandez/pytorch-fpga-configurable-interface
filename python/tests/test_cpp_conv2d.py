import numpy as np
import torch

import cpp_kernels


def validate_conv2d(
    name,
    input_torch,
    conv_torch,
):
    with torch.no_grad():
        output_torch = conv_torch(
            input_torch
        )

    input_numpy = (
        input_torch
        .detach()
        .numpy()
    )

    weight_numpy = (
        conv_torch.weight
        .detach()
        .numpy()
    )

    if conv_torch.bias is None:
        bias_numpy = None
    else:
        bias_numpy = (
            conv_torch.bias
            .detach()
            .numpy()
        )

    output_height = output_torch.shape[2]
    output_width = output_torch.shape[3]

    stride_height, stride_width = (
        conv_torch.stride
    )

    padding_height, padding_width = (
        conv_torch.padding
    )

    output_cpp = cpp_kernels.conv2d_forward(
        input_numpy,
        weight_numpy,
        bias_numpy,
        output_height,
        output_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
    )

    output_reference = (
        output_torch
        .detach()
        .numpy()
    )

    error = output_cpp - output_reference

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    max_absolute_error = np.max(
        np.abs(error)
    )

    print(f"{name}:")
    print("  Forma C++:", output_cpp.shape)
    print(
        "  Forma PyTorch:",
        output_reference.shape,
    )
    print("  RMSE:", rmse)
    print(
        "  Error absoluto máximo:",
        max_absolute_error,
    )

    assert output_cpp.shape == output_reference.shape

    assert np.allclose(
        output_cpp,
        output_reference,
        atol=1e-5,
        rtol=1e-4,
    )

    print("  PASSED")


torch.manual_seed(42)

input_torch = torch.randn(
    2,
    3,
    8,
    9,
    dtype=torch.float32,
)

conv_with_bias = torch.nn.Conv2d(
    in_channels=3,
    out_channels=4,
    kernel_size=(3, 2),
    stride=(2, 1),
    padding=(1, 0),
    dilation=1,
    groups=1,
    bias=True,
)

validate_conv2d(
    name="Conv2d con bias",
    input_torch=input_torch,
    conv_torch=conv_with_bias,
)

conv_without_bias = torch.nn.Conv2d(
    in_channels=3,
    out_channels=4,
    kernel_size=(3, 2),
    stride=(2, 1),
    padding=(1, 0),
    dilation=1,
    groups=1,
    bias=False,
)

validate_conv2d(
    name="Conv2d sin bias",
    input_torch=input_torch,
    conv_torch=conv_without_bias,
)

print("test_cpp_conv2d: PASSED")


