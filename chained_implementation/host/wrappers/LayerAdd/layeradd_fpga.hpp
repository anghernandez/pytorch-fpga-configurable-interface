#ifndef LAYERADD_FPGA_HPP
#define LAYERADD_FPGA_HPP

#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_kernel.h>

// ------------------------------------------------------------
// Lanzamiento asíncrono
// ------------------------------------------------------------
// El llamador conserva los tres buffers hasta la finalización.
// No reserva memoria ni realiza transferencias CPU <-> FPGA.
xrt::run layeradd_fpga_forward_async(
    xrt::bo& input1_bo,
    xrt::bo& input2_bo,
    xrt::bo& output_bo,
    int size,
    float alpha
);

// ------------------------------------------------------------
// Ejecución síncrona
// ------------------------------------------------------------
// Lanza LayerAdd y espera su finalización.
void layeradd_fpga_forward(
    xrt::bo& input1_bo,
    xrt::bo& input2_bo,
    xrt::bo& output_bo,
    int size,
    float alpha
);

#endif // LAYERADD_FPGA_HPP