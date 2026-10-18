#ifndef RELU6_FPGA_HPP
#define RELU6_FPGA_HPP

#include <xrt/xrt/xrt_bo.h>

// Ejecuta ReLU6 sobre buffers XRT administrados por el llamador.
// No reserva buffers ni realiza transferencias CPU <-> FPGA.
void relu6_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& output_bo,
    int size
);

#endif // RELU6_FPGA_HPP

