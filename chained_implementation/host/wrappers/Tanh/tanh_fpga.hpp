#ifndef TANH_FPGA_HPP
#define TANH_FPGA_HPP

#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_kernel.h>

// Lanza Tanh sin esperar su finalización.
// El llamador conserva los buffers hasta que termine la ejecución.
xrt::run tanh_fpga_forward_async(
    xrt::bo& input_bo,
    xrt::bo& output_bo,
    int size
);

// Ejecuta Tanh y espera su finalización.
// No reserva buffers ni realiza transferencias CPU <-> dispositivo.
void tanh_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& output_bo,
    int size
);

#endif // TANH_FPGA_HPP