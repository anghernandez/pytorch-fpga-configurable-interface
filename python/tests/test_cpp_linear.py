import numpy as np
import torch

import cpp_kernels


torch.manual_seed(42)

batch_size = 4
in_features = 8
out_features = 5

input_torch = torch.randn(
    batch_size,
    in_features,
    dtype=torch.float32,
)

linear_torch = torch.nn.Linear(
    in_features,
    out_features,
    bias=True,
)

with torch.no_grad():
    output_torch = linear_torch(
        input_torch
    )

input_numpy = (
    input_torch
    .detach()
    .numpy()
)

weight_numpy = (
    linear_torch.weight
    .detach()
    .numpy()
)

bias_numpy = (
    linear_torch.bias
    .detach()
    .numpy()
)

output_cpp = cpp_kernels.linear_forward(
    input_numpy,
    weight_numpy,
    bias_numpy,
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

print("Forma C++:", output_cpp.shape)
print("Forma PyTorch:", output_reference.shape)
print("RMSE:", rmse)
print("Error absoluto máximo:", max_absolute_error)

assert output_cpp.shape == output_reference.shape

assert np.allclose(
    output_cpp,
    output_reference,
    atol=1e-6,
    rtol=1e-5,
)

print("Linear con bias: PASSED")


linear_without_bias = torch.nn.Linear(
    in_features,
    out_features,
    bias=False,
)

with torch.no_grad():
    output_without_bias_torch = (
        linear_without_bias(input_torch)
    )

output_without_bias_cpp = (
    cpp_kernels.linear_forward(
        input_numpy,
        (
            linear_without_bias.weight
            .detach()
            .numpy()
        ),
        None,
    )
)

assert np.allclose(
    output_without_bias_cpp,
    (
        output_without_bias_torch
        .detach()
        .numpy()
    ),
    atol=1e-6,
    rtol=1e-5,
)

print("Linear sin bias: PASSED")
print("test_cpp_linear: PASSED")
