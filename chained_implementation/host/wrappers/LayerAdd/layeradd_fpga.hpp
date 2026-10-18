#ifndef LAYERADD_FPGA_HPP
#define LAYERADD_FPGA_HPP

#include <xrt/xrt/xrt_bo.h>

// Ejecuta LayerAdd con tres buffers externos y parametros escalares.
// La administracion y sincronizacion de buffers corresponden al llamador.
void layeradd_fpga_forward(
    xrt::bo& input1_bo,
    xrt::bo& input2_bo,
    xrt::bo& output_bo,
    int size,
    float alpha
);

#endif // LAYERADD_FPGA_HPP