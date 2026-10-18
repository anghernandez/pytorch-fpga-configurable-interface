#ifndef AVGPOOL2D_FPGA_HPP
#define AVGPOOL2D_FPGA_HPP

#include <xrt/xrt/xrt_bo.h>

// Ejecuta AvgPool2D sobre buffers XRT administrados externamente.
// No asigna buffers ni sincroniza datos entre CPU y FPGA.
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