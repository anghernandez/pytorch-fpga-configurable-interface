#include "layeradd_fpga.hpp"
#include "fpga_runtime.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <xrt/xrt/xrt_kernel.h>

void layeradd_fpga_forward(
    xrt::bo& input1_bo,
    xrt::bo& input2_bo,
    xrt::bo& output_bo,
    int size,
    float alpha
)
{
    // ============================================================
    // 1. VALIDACION DE PARAMETROS ESCALARES
    // ============================================================
    if (size <= 0) {
        throw std::invalid_argument("size debe ser mayor que cero");
    }
    if (!std::isfinite(alpha)) {
        throw std::invalid_argument("alpha debe ser finito");
    }

    // ============================================================
    // 2. VALIDACION DE LA CAPACIDAD DE LOS BUFFERS EXTERNOS
    // ============================================================
    const std::size_t bytes =
        static_cast<std::size_t>(size) * sizeof(float);

    if (input1_bo.size() < bytes ||
        input2_bo.size() < bytes ||
        output_bo.size() < bytes) {
        throw std::invalid_argument("Buffers XRT insuficientes para LayerAdd");
    }

    // ============================================================
    // 3. OBTENCION DEL RUNTIME Y DEL KERNEL XRT
    // ============================================================
    auto& runtime = FpgaRuntime::instance();
    xrt::kernel kernel(runtime.device(), runtime.uuid(), "layer_add_forward");

    // ============================================================
    // 4. EJECUCION CON BUFFERS EXTERNOS
    // ============================================================
    // Los dos buffers de entrada ya contienen datos accesibles por FPGA.
    // Ninguna entrada ni salida se copia hacia o desde CPU aqui.
    auto run = kernel(input1_bo, input2_bo, output_bo, size, alpha);

    // ============================================================
    // 5. FINALIZACION SIN TRANSFERENCIA FPGA -> CPU
    // ============================================================
    run.wait();
    // El llamador conserva la propiedad de los buffers de entrada y salida.
    // El runtime futuro debera conservar las entradas aun requeridas
    // por conexiones residuales hasta que dejen de utilizarse.
}