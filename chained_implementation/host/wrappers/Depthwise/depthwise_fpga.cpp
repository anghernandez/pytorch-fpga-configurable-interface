#include "depthwise_fpga.hpp"
#include "fpga_runtime.hpp"

#include <cstddef>
#include <stdexcept>
#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>

void depthwise_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& output_bo,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool use_bias
)
{
    // ============================================================
    // 1. VALIDACION DE DIMENSIONES Y PARAMETROS
    // ============================================================
    if (batch_size <= 0 || channels <= 0 ||
        input_height <= 0 || input_width <= 0 ||
        output_height <= 0 || output_width <= 0 ||
        kernel_height <= 0 || kernel_width <= 0 ||
        stride_height <= 0 || stride_width <= 0 ||
        padding_height < 0 || padding_width < 0) {
        throw std::invalid_argument("Dimensiones, stride o padding invalidos en DepthwiseConv2D");
    }
    const long long expected_h =
        (static_cast<long long>(input_height) + 2LL * padding_height - kernel_height)
        / stride_height + 1;
    const long long expected_w =
        (static_cast<long long>(input_width) + 2LL * padding_width - kernel_width)
        / stride_width + 1;
    if (expected_h != output_height || expected_w != output_width)
        throw std::invalid_argument("Dimensiones de salida incompatibles con DepthwiseConv2D");

    const std::size_t input_count = static_cast<std::size_t>(batch_size)
        * channels * input_height * input_width;
    const std::size_t weight_count = static_cast<std::size_t>(channels)
        * kernel_height * kernel_width;
    const std::size_t bias_count = static_cast<std::size_t>(channels);
    const std::size_t output_count = static_cast<std::size_t>(batch_size)
        * channels * output_height * output_width;

    // ============================================================
    // 2. VALIDACION DE BUFFERS EXTERNOS
    // ============================================================
    // bias_bo debe existir aunque use_bias=false: es argumento de HLS.
    if (input_bo.size() < input_count * sizeof(float) ||
        weight_bo.size() < weight_count * sizeof(float) ||
        bias_bo.size() < bias_count * sizeof(float) ||
        output_bo.size() < output_count * sizeof(float)) {
        throw std::invalid_argument("Buffers XRT insuficientes para DepthwiseConv2D");
    }

    // ============================================================
    // 3. RUNTIME COMPARTIDO Y KERNEL XRT
    // ============================================================
    auto& runtime = FpgaRuntime::instance();
    xrt::kernel kernel(runtime.device(), runtime.uuid(), "depthwise_conv2d_forward");

    // ============================================================
    // 4. EJECUCION SOBRE LOS BUFFERS PROPORCIONADOS
    // ============================================================
    // No se asigna memoria ni se transfieren tensores en el wrapper.
    // La gestion de pesos, bias, entradas y salidas corresponde al llamador.
    auto run = kernel(input_bo, weight_bo, bias_bo, output_bo,
        batch_size, channels, input_height, input_width,
        output_height, output_width, kernel_height, kernel_width,
        stride_height, stride_width, padding_height, padding_width, use_bias);

    // ============================================================
    // 5. FINALIZACION SIN COPIAR LA SALIDA A CPU
    // ============================================================
    run.wait();
    // output_bo permanece accesible para otros kernels.
}