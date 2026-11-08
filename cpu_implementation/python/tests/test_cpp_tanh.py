import numpy as np
import torch

import cpp_kernels


input_numpy = np.array(
    [
        -100.0,
        -10.0,
        -2.0,
        -1.0,
        -0.25,
        0.0,
        0.25,
        1.0,
        2.0,
        10.0,
        100.0,
    ],
    dtype=np.float32,
)

output_cpp = cpp_kernels.tanh_forward(
    input_numpy
)

output_torch = torch.tanh(
    torch.from_numpy(input_numpy)
).numpy()

error = output_cpp - output_torch

rmse = np.sqrt(
    np.mean(error ** 2)
)

max_absolute_error = np.max(
    np.abs(error)
)

print("Salida C++:    ", output_cpp)
print("Salida PyTorch:", output_torch)
print("RMSE:", rmse)
print("Error absoluto máximo:", max_absolute_error)

assert output_cpp.shape == output_torch.shape

assert np.allclose(
    output_cpp,
    output_torch,
    atol=1e-6,
    rtol=1e-5,
)

print("test_cpp_tanh: PASSED")
