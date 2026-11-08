import numpy as np
import torch
import torch.nn.functional as functional

import cpp_kernels


def validate_case(
    name,
    input_torch,
    kernel_size,
    stride,
    padding,
    count_include_pad,
    divisor_override=None,
):
    output_torch = functional.avg_pool2d(
        input_torch,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=False,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    output_height = output_torch.shape[2]
    output_width = output_torch.shape[3]

    output_cpp = cpp_kernels.avg_pool2d_forward(
        input_torch.detach().numpy(),
        output_height,
        output_width,
        kernel_size[0],
        kernel_size[1],
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        count_include_pad,
        divisor_override,
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
    print("  Forma:", output_cpp.shape)
    print("  RMSE:", rmse)
    print(
        "  Error absoluto máximo:",
        max_absolute_error,
    )

    assert output_cpp.shape == output_reference.shape

    assert np.allclose(
        output_cpp,
        output_reference,
        atol=1e-6,
        rtol=1e-5,
    )

    print("  PASSED")


torch.manual_seed(42)

input_torch = torch.randn(
    2,
    3,
    6,
    7,
    dtype=torch.float32,
)

validate_case(
    name="AvgPool2d incluyendo padding",
    input_torch=input_torch,
    kernel_size=(2, 3),
    stride=(2, 2),
    padding=(0, 1),
    count_include_pad=True,
)

validate_case(
    name="AvgPool2d excluyendo padding",
    input_torch=input_torch,
    kernel_size=(2, 3),
    stride=(2, 2),
    padding=(0, 1),
    count_include_pad=False,
)

validate_case(
    name="AvgPool2d con divisor_override",
    input_torch=input_torch,
    kernel_size=(2, 3),
    stride=(2, 2),
    padding=(0, 1),
    count_include_pad=True,
    divisor_override=5,
)

print("test_cpp_avg_pool2d: PASSED")

