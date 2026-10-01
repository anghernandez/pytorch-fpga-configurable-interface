#ifndef RELU6_FPGA_HPP
#define RELU6_FPGA_HPP

#include <string>

void relu6_fpga_forward(
    const float* input,
    float* output,
    int size
   // const std::string& xclbin_path
);

#endif
