#ifndef RELU6_FPGA_HPP
#define RELU6_FPGA_HPP

#include <string>
#include <xrt/xrt/xrt_kernel.h>
#include "fpga_runtime.hpp"

// Interfaz pública: CPU → FPGA → CPU
void relu6_fpga_forward(
    const float* input,
    float* output,
    int size
   // const std::string& xclbin_path
);


#endif