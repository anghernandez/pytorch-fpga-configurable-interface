#ifndef CONV2D_FPGA_HPP
#define CONV2D_FPGA_HPP

#include <xrt/xrt/xrt_bo.h>

// Ejecuta Conv2D con buffers XRT creados y gestionados externamente.
// No crea buffers ni efectúa sincronizaciones CPU <-> FPGA.
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