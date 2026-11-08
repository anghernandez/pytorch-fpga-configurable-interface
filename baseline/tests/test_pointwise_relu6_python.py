import numpy as np
import fpga_buffer

# ============================================================
# Datos de prueba
# ============================================================

# N=1, C_in=2, H=2, W=2
x = np.array(
    [
        [
            [[1, 2],
             [3, 4]],

            [[5, 6],
             [7, 8]]
        ]
    ],
    dtype=np.float32
)

# Pointwise: C_in=2 -> C_out=2
# weight shape: [2, 2, 1, 1]
weight = np.array(
    [
        [[[1]], [[1]]],
        [[[2]], [[1]]]
    ],
    dtype=np.float32
)

bias = np.array([0, 1], dtype=np.float32)

# ============================================================
# CPU reference
# ============================================================

expected_pointwise = np.empty((1, 2, 2, 2), dtype=np.float32)

expected_pointwise[0, 0] = x[0, 0] + x[0, 1]
expected_pointwise[0, 1] = 2 * x[0, 0] + x[0, 1] + 1

expected_relu6 = np.clip(expected_pointwise, 0, 6)

# ============================================================
# FPGA
# ============================================================

print("Ejecutando Pointwise en FPGA...")

x_fpga = fpga_buffer.from_numpy(x)
w_fpga = fpga_buffer.pointwise_weight_from_numpy(weight)
b_fpga = fpga_buffer.pointwise_bias_from_numpy(bias)

x_fpga = fpga_buffer.pointwise_forward(
    x_fpga,
    w_fpga,
    b_fpga
)

print("Pointwise terminado.")

# IMPORTANTE:
# Aquí NO hacemos to_numpy().
# El resultado sigue en un FpgaTensor/BO de FPGA.

print("Ejecutando ReLU6 directamente sobre el buffer FPGA...")

x_fpga = fpga_buffer.relu6_forward(x_fpga)

print("ReLU6 terminado.")

# ÚNICA transferencia FPGA -> CPU:
result = x_fpga.to_numpy()

# ============================================================
# Comparación
# ============================================================

print("\nResultado FPGA:")
print(result)

print("\nResultado esperado:")
print(expected_relu6)

max_error = np.max(np.abs(result - expected_relu6))

print("\nError máximo:", max_error)

if max_error < 1e-5:
    print("RESULTADO: PASS")
else:
    print("RESULTADO: FAIL")
    raise SystemExit(1)
