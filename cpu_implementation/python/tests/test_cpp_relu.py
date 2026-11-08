import numpy as np
import torch

import cpp_kernels


rng = np.random.default_rng(seed=42)

input_numpy = rng.normal(
    size=(2, 3, 4, 5)
).astype(np.float32)

output_cpp = cpp_kernels.relu_forward(
    input_numpy
)

input_torch = torch.from_numpy(
    input_numpy
)

output_torch = torch.relu(
    input_torch
).numpy()

error = output_cpp - output_torch

rmse = np.sqrt(
    np.mean(error ** 2)
)

max_absolute_error = np.max(
    np.abs(error)
)

print("Forma de entrada:", input_numpy.shape)
print("Forma C++:", output_cpp.shape)
print("Forma PyTorch:", output_torch.shape)
print("RMSE:", rmse)
print("Error absoluto máximo:", max_absolute_error)

assert output_cpp.shape == output_torch.shape

assert np.allclose(
    output_cpp,
    output_torch,
    atol=1e-6,
    rtol=1e-5,
)

print("test_cpp_relu: PASSED")

