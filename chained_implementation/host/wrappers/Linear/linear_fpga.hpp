#ifndef LINEAR_FPGA_HPP
#define LINEAR_FPGA_HPP

#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_kernel.h>

// ------------------------------------------------------------
// Lanzamiento asíncrono
// ------------------------------------------------------------
// El llamador conserva los buffers hasta que termine la ejecución.
// bias_bo debe ser válido incluso cuando use_bias=false.
// No reserva memoria ni realiza transferencias CPU <-> dispositivo.
xrt::run linear_fpga_forward_async(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& output_bo,
    int batch_size,
    int in_features,
    int out_features,
    bool use_bias
);

// ------------------------------------------------------------
// Ejecución síncrona
// ------------------------------------------------------------
// Lanza Linear y espera su finalización.
void linear_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& output_bo,
    int batch_size,
    int in_features,
    int out_features,
    bool use_bias
);

#endif // LINEAR_FPGA_HPP