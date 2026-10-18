#include "batchnorm2d_fpga.hpp"
#include "fpga_runtime.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>

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
)
{
    // ============================================================
    // 1. VALIDACION DE PARAMETROS
    // ============================================================
    if (batch_size <= 0 || channels <= 0 ||
        input_height <= 0 || input_width <= 0) {
        throw std::invalid_argument("Las dimensiones deben ser mayores que cero");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument("eps debe ser positivo y finito");
    }

    // ============================================================
    // 2. VERIFICACION DE CAPACIDAD DE BUFFERS EXTERNOS
    // ============================================================
    const std::size_t elements =
        static_cast<std::size_t>(batch_size) *
        static_cast<std::size_t>(channels) *
        static_cast<std::size_t>(input_height) *
        static_cast<std::size_t>(input_width);
    const std::size_t data_bytes = elements * sizeof(float);
    const std::size_t param_bytes =
        static_cast<std::size_t>(channels) * sizeof(float);

    // Los seis argumentos de memoria existen en la firma HLS,
    // incluso si affine o use_bias son falsos.
    if (input_bo.size() < data_bytes || output_bo.size() < data_bytes ||
        weight_bo.size() < param_bytes || bias_bo.size() < param_bytes ||
        running_mean_bo.size() < param_bytes ||
        running_var_bo.size() < param_bytes) {
        throw std::invalid_argument("Buffers XRT insuficientes para BatchNorm2D");
    }

    // ============================================================
    // 3. OBTENCION DEL RUNTIME Y DEL KERNEL
    // ============================================================
    auto& runtime = FpgaRuntime::instance();
    xrt::kernel kernel(runtime.device(), runtime.uuid(), "batchnorm2d_forward");

    // ============================================================
    // 4. EJECUCION CON BUFFERS EXTERNOS
    // ============================================================
    // Los parametros y entradas deben haber sido sincronizados hacia
    // el dispositivo por el llamador. No se crean ni copian buffers aqui.
    auto run = kernel(
        input_bo, weight_bo, bias_bo, running_mean_bo, running_var_bo,
        output_bo, batch_size, channels, input_height, input_width,
        eps, affine, use_bias
    );

    // ============================================================
    // 5. FINALIZACION SIN RECUPERAR LOS DATOS EN CPU
    // ============================================================
    run.wait();
    // output_bo permanece disponible para otra operacion FPGA.
}