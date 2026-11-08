#ifndef AVGPOOL2D_FPGA_HPP
#define AVGPOOL2D_FPGA_HPP

#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_kernel.h>

// ------------------------------------------------------------
// Lanzamiento asíncrono
// ------------------------------------------------------------
// El llamador conserva los buffers hasta que termine la ejecución.
// No reserva memoria ni realiza transferencias CPU <-> FPGA.
xrt::run avgpool2d_fpga_forward_async(
    xrt::bo& input_bo,
    xrt::bo& output_bo,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool count_include_pad,
    bool use_divisor_override,
    int divisor_override
);

// ------------------------------------------------------------
// Ejecución síncrona
// ------------------------------------------------------------
// Lanza AvgPool2D y espera su finalización.
void avgpool2d_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& output_bo,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool count_include_pad,
    bool use_divisor_override,
    int divisor_override
);

#endif // AVGPOOL2D_FPGA_HPP