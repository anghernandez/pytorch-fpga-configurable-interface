#ifndef BATCHNORM2D_FPGA_HPP
#define BATCHNORM2D_FPGA_HPP

#include <string>

void batchnorm2d_fpga_forward(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    float eps,
    bool affine,
    bool use_bias
   // const std::string& xclbin_path
);

#endif