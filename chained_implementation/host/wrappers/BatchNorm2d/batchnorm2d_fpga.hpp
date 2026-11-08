#ifndef BATCHNORM2D_FPGA_HPP
#define BATCHNORM2D_FPGA_HPP

#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_kernel.h>

// ------------------------------------------------------------
// Lanzamiento asíncrono en modo inferencia
// ------------------------------------------------------------
// El llamador conserva todos los buffers hasta la finalización.
// No reserva memoria ni realiza transferencias CPU <-> FPGA.
xrt::run batchnorm2d_fpga_forward_async(
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

// ------------------------------------------------------------
// Ejecución síncrona en modo inferencia
// ------------------------------------------------------------
// Lanza BatchNorm2D y espera su finalización.
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