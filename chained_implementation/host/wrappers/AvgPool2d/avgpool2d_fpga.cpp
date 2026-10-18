#include "avgpool2d_fpga.hpp"
#include "fpga_runtime.hpp"

#include <cstddef>
#include <stdexcept>
#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>

void avgpool2d_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& output_bo,
    int batch_size,
    int input_channels,
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
    bool count_include_pad,
    bool use_divisor_override,
    int divisor_override
)
{
    // ============================================================
    // 1. VALIDACIÓN DE DIMENSIONES Y CONFIGURACIÓN
    // ============================================================
    if (batch_size <= 0 || input_channels <= 0 ||
        input_height <= 0 || input_width <= 0 ||
        output_height <= 0 || output_width <= 0 ||
        kernel_height <= 0 || kernel_width <= 0 ||
        stride_height <= 0 || stride_width <= 0) {
        throw std::invalid_argument("Dimensiones, kernel y stride deben ser positivos");
    }
    if (padding_height < 0 || padding_width < 0) {
        throw std::invalid_argument("El padding no puede ser negativo");
    }
    if (use_divisor_override && divisor_override <= 0) {
        throw std::invalid_argument("divisor_override debe ser positivo cuando se utiliza");
    }

    // Se utiliza aritmética de mayor rango para validar las dimensiones.
    const auto numerator_h = static_cast<long long>(input_height)
                           + 2LL * padding_height - kernel_height;
    const auto numerator_w = static_cast<long long>(input_width)
                           + 2LL * padding_width - kernel_width;
    if (numerator_h < 0 || numerator_w < 0 ||
        numerator_h / stride_height + 1 != output_height ||
        numerator_w / stride_width + 1 != output_width) {
        throw std::invalid_argument("Las dimensiones de salida no coinciden con AvgPool2D");
    }

    // ============================================================
    // 2. VALIDACIÓN DE CAPACIDAD DE LOS BUFFERS EXTERNOS
    // ============================================================
    const std::size_t input_elements =
        static_cast<std::size_t>(batch_size) * input_channels * input_height * input_width;
    const std::size_t output_elements =
        static_cast<std::size_t>(batch_size) * input_channels * output_height * output_width;

    if (input_bo.size() < input_elements * sizeof(float) ||
        output_bo.size() < output_elements * sizeof(float)) {
        throw std::invalid_argument("Los buffers XRT no tienen capacidad suficiente para AvgPool2D");
    }

    // ============================================================
    // 3. OBTENCIÓN DEL RUNTIME Y DEL KERNEL XRT
    // ============================================================
    auto& runtime = FpgaRuntime::instance();
    xrt::device& device = runtime.device();
    xrt::kernel kernel(device, runtime.uuid(), "avg_pool2d_forward");

    // ============================================================
    // 4. EJECUCIÓN CON BUFFERS RECIBIDOS (SIN COPIAS A CPU)
    // ============================================================
    // input_bo puede contener la salida de cualquier kernel anterior.
    // output_bo conserva los resultados para otro kernel posterior.
    auto run = kernel(
        input_bo, output_bo,
        batch_size, input_channels, input_height, input_width,
        output_height, output_width,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_height, padding_width,
        count_include_pad, use_divisor_override, divisor_override
    );

    // ============================================================
    // 5. FINALIZACIÓN (SIN SINCRONIZACIÓN FPGA -> CPU)
    // ============================================================
    run.wait();
}