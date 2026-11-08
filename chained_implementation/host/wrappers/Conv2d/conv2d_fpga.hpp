#ifndef CONV2D_FPGA_HPP
#define CONV2D_FPGA_HPP

#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_kernel.h>

// ------------------------------------------------------------
// Lanzamiento asíncrono
// ------------------------------------------------------------
// El llamador conserva todos los buffers hasta la finalización.
// No reserva memoria ni realiza transferencias CPU <-> FPGA.
xrt::run conv2d_fpga_forward_async(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& output_bo,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool use_bias
);

// ------------------------------------------------------------
// Ejecución síncrona
// ------------------------------------------------------------
void conv2d_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& output_bo,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool use_bias
);

#endif // CONV2D_FPGA_HPP