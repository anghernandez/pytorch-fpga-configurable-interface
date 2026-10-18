#ifndef TANH_FPGA_HPP
#define TANH_FPGA_HPP

// Tipo de buffer administrado por XRT.
#include <xrt/xrt/xrt_bo.h>

// Ejecuta Tanh con buffers previamente asignados por el llamador.
// No reserva buffers ni realiza transferencias CPU <-> dispositivo.
void tanh_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& output_bo,
    int size
);

#endif

