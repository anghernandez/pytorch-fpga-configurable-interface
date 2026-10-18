#include "conv2d_fpga.hpp"
#include "fpga_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>

void conv2d_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& output_bo,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
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
    // 1. VALIDACIÓN DE DIMENSIONES Y PARÁMETROS DE CONVOLUCIÓN
    // ============================================================
    if (batch_size <= 0 || input_channels <= 0 ||
        input_height <= 0 || input_width <= 0 || output_channels <= 0 ||
        kernel_height <= 0 || kernel_width <= 0 ||
        stride_height <= 0 || stride_width <= 0 ||
        padding_height < 0 || padding_width < 0) {
        throw std::invalid_argument("Dimensiones, kernel, stride o padding invalidos");
    }

    // Se calcula con enteros amplios para evitar desbordamientos intermedios.
    const std::int64_t numerator_h =
        static_cast<std::int64_t>(input_height) + 2LL * padding_height - kernel_height;
    const std::int64_t numerator_w =
        static_cast<std::int64_t>(input_width) + 2LL * padding_width - kernel_width;
    if (numerator_h < 0 || numerator_w < 0) {
        throw std::invalid_argument("La convolucion produce dimensiones invalidas");
    }
    const std::int64_t expected_h = numerator_h / stride_height + 1;
    const std::int64_t expected_w = numerator_w / stride_width + 1;
    if (output_height <= 0 || output_width <= 0 ||
        static_cast<std::int64_t>(output_height) != expected_h ||
        static_cast<std::int64_t>(output_width) != expected_w) {
        throw std::invalid_argument("Las dimensiones de salida son incorrectas");
    }

    // ============================================================
    // 2. CÁLCULO DE CAPACIDAD Y VALIDACIÓN DE BUFFERS EXTERNOS
    // ============================================================
    // Los tensores usan disposición NCHW y pesos [Cout, Cin, Kh, Kw].
    const std::size_t input_count =
        static_cast<std::size_t>(batch_size) * input_channels * input_height * input_width;
    const std::size_t weight_count =
        static_cast<std::size_t>(output_channels) * input_channels * kernel_height * kernel_width;
    const std::size_t bias_count = static_cast<std::size_t>(output_channels);
    const std::size_t output_count =
        static_cast<std::size_t>(batch_size) * output_channels * output_height * output_width;

    // La firma HLS siempre contiene bias_bo, incluso con use_bias=false.
    // En tal caso, su contenido no debe afectar el resultado del kernel.
    if (input_bo.size() < input_count * sizeof(float) ||
        weight_bo.size() < weight_count * sizeof(float) ||
        bias_bo.size() < bias_count * sizeof(float) ||
        output_bo.size() < output_count * sizeof(float)) {
        throw std::invalid_argument("Buffers XRT insuficientes para Conv2D");
    }

    // ============================================================
    // 3. OBTENCIÓN DEL RUNTIME Y KERNEL
    // ============================================================
    auto& runtime = FpgaRuntime::instance();
    xrt::device& device = runtime.device();
    xrt::kernel kernel(device, runtime.uuid(), "conv2d_forward");

    // ============================================================
    // 4. EJECUCIÓN CON BUFFERS PROPORCIONADOS POR EL LLAMADOR
    // ============================================================
    // No se asigna memoria, no se mapean buffers y no se copian datos.
    // El llamador asegura que entrada, pesos y bias estén disponibles
    // en el dispositivo antes de la ejecución.
    auto run = kernel(
        input_bo, weight_bo, bias_bo, output_bo,
        batch_size, input_channels, input_height, input_width,
        output_channels, output_height, output_width,
        kernel_height, kernel_width, stride_height, stride_width,
        padding_height, padding_width, use_bias
    );

    // ============================================================
    // 5. ESPERA SIN RECUPERACIÓN INTERMEDIA A CPU
    // ============================================================
    run.wait();
    // output_bo queda disponible para el siguiente kernel.
}