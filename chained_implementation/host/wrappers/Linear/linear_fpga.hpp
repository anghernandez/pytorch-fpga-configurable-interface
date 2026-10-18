#ifndef LINEAR_FPGA_HPP
#define LINEAR_FPGA_HPP

#include <xrt/xrt/xrt_bo.h>

// Ejecuta Linear utilizando buffers XRT creados y administrados externamente.
// El llamador prepara la entrada, los pesos y el bias, incluso cuando
// use_bias=false (el kernel sigue recibiendo un argumento de buffer).
// No realiza transferencias CPU <-> dispositivo ni asigna memoria.
void linear_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& output_bo,
    int batch_size,
    int in_features,
    int out_features,
    bool use_bias
);

#endif

