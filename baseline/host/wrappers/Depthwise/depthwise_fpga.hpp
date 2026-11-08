#ifndef DEPTHWISE_FPGA_HPP
#define DEPTHWISE_FPGA_HPP

#include <string>

void depthwise_fpga_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
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
       // const std::string& xclbin_path
);

#endif
