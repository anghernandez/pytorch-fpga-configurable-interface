#ifndef LAYERADD_FPGA_HPP
#define LAYERADD_FPGA_HPP

#include <string>

void layeradd_fpga_forward(
    const float* input1,
    const float* input2,
    float* output,
    int size,
    float alpha
   //const std::string& xclbin_path
);

#endif