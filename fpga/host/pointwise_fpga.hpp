#ifndef POINTWISE_FPGA_HPP
#define POINTWISE_FPGA_HPP

#include <string>

void pointwise_fpga_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    bool use_bias
   // const std::string& xclbin_path
);

#endif
