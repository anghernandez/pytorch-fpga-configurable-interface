#ifndef POINTWISE_FPGA_HPP
#define POINTWISE_FPGA_HPP
#include <xrt/xrt/xrt_bo.h>
// Buffers externos gestionados por el llamador/runtime.
void pointwise_fpga_forward(xrt::bo& input_bo, xrt::bo& weight_bo,
    xrt::bo& bias_bo, xrt::bo& output_bo, int batch_size,
    int input_channels, int input_height, int input_width,
    int output_channels, bool use_bias);
#endif