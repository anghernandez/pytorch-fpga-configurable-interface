#ifndef BATCHNORM2D_FPGA_HPP
#define BATCHNORM2D_FPGA_HPP

#include <xrt/xrt/xrt_bo.h>

// Ejecuta BatchNorm2D en modo inferencia sobre buffers XRT externos.
// El llamador administra la reserva, carga, sincronizacion y vida de los buffers.
void batchnorm2d_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& running_mean_bo,
    xrt::bo& running_var_bo,
    xrt::bo& output_bo,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    float eps,
    bool affine,
    bool use_bias
);

#endif // BATCHNORM2D_FPGA_HPP