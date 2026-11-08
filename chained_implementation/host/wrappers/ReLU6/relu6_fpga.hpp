#ifndef RELU6_FPGA_HPP
#define RELU6_FPGA_HPP

#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_kernel.h>

// Lanza ReLU6 y devuelve el control sin esperar su finalización.
// El llamador debe conservar los buffers hasta que termine la ejecución.
xrt::run relu6_fpga_forward_async(
    xrt::bo& input_bo,
    xrt::bo& output_bo,
    int size
);

// Ejecuta ReLU6 y espera su finalización.
// No reserva buffers ni realiza transferencias CPU <-> FPGA.
void relu6_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& output_bo,
    int size
);


// Lanza Pointwise sin esperar su finalización.
// El llamador conserva todos los buffers hasta que termine.
xrt::run pointwise_fpga_forward_async(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& output_bo,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    bool use_bias
);

#endif // RELU6_FPGA_HPP