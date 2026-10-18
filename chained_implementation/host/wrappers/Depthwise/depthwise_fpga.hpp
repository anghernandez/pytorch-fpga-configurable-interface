#ifndef DEPTHWISE_FPGA_HPP
#define DEPTHWISE_FPGA_HPP

#include <xrt/xrt/xrt_bo.h>

// Ejecuta DepthwiseConv2D usando buffers externos administrados por el llamador.
// No realiza transferencias entre CPU y FPGA.
void depthwise_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& output_bo,
    int batch_size,
    int channels,
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
    bool use_bias
);

#endif // DEPTHWISE_FPGA_HPP