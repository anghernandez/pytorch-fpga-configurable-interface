#include "pointwise_fpga.hpp"
#include "fpga_runtime.hpp"
#include <cstddef>
#include <stdexcept>
#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>

void pointwise_fpga_forward(
    xrt::bo& input_bo, xrt::bo& weight_bo,
    xrt::bo& bias_bo, xrt::bo& output_bo,
    int batch_size, int input_channels, int input_height,
    int input_width, int output_channels, bool use_bias)
{
    // ============================================================
    // 1. VALIDACIONES DE DIMENSIONES
    // ============================================================
    if (batch_size <= 0 || input_channels <= 0 || input_height <= 0 ||
        input_width <= 0 || output_channels <= 0)
        throw std::invalid_argument("Las dimensiones deben ser mayores que cero");

    // Tensor NCHW y pesos de convolucion 1x1 [C_out, C_in].
    const std::size_t input_count = static_cast<std::size_t>(batch_size)
        * input_channels * input_height * input_width;
    const std::size_t weight_count = static_cast<std::size_t>(output_channels)
        * input_channels;
    const std::size_t bias_count = static_cast<std::size_t>(output_channels);
    const std::size_t output_count = static_cast<std::size_t>(batch_size)
        * output_channels * input_height * input_width;

    // ============================================================
    // 2. VALIDACION DE CAPACIDAD DE BUFFERS EXTERNOS
    // ============================================================
    // bias_bo debe existir aun cuando use_bias=false (firma HLS).
    if (input_bo.size() < input_count * sizeof(float) ||
        weight_bo.size() < weight_count * sizeof(float) ||
        bias_bo.size() < bias_count * sizeof(float) ||
        output_bo.size() < output_count * sizeof(float))
        throw std::invalid_argument("Buffers XRT insuficientes para PointwiseConv2D");

    // ============================================================
    // 3. OBTENCION DEL RUNTIME Y DEL KERNEL XRT
    // ============================================================
    auto& runtime = FpgaRuntime::instance();
    xrt::kernel kernel(runtime.device(), runtime.uuid(), "pointwise_conv2d_forward");

    // ============================================================
    // 4. EJECUCION CON BUFFERS SUMINISTRADOS EXTERNAMENTE
    // ============================================================
    // La asignacion y sincronizacion CPU->FPGA corresponden al llamador.
    auto run = kernel(input_bo, weight_bo, bias_bo, output_bo,
        batch_size, input_channels, input_height, input_width,
        output_channels, use_bias);

    // ============================================================
    // 5. ESPERA SIN TRANSFERENCIA DE RESULTADOS A CPU
    // ============================================================
    run.wait();
    // output_bo queda disponible para la siguiente operacion.
}