#ifndef POINTWISE_FPGA_HPP
#define POINTWISE_FPGA_HPP

#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_kernel.h>

// ------------------------------------------------------------
// Lanzamiento asíncrono
// ------------------------------------------------------------
// El llamador conserva todos los buffers hasta la finalización.
// No reserva memoria ni realiza transferencias CPU <-> FPGA.
xrt::run pointwise_fpga_forward_async(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& output_bo,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    bool use_bias
);

// ------------------------------------------------------------
// Ejecución síncrona
// ------------------------------------------------------------
// Lanza Pointwise y espera su finalización.
void pointwise_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& output_bo,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    bool use_bias
);

#endif // POINTWISE_FPGA_HPP