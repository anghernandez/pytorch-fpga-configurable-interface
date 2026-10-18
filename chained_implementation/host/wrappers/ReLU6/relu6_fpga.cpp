#include "relu6_fpga.hpp"
#include "fpga_runtime.hpp"

#include <cstddef>
#include <stdexcept>

#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>

void relu6_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& output_bo,
    int size
)
{
    // ============================================================
    // 1. VALIDACIÓN DE PARÁMETROS Y CAPACIDAD DE LOS BUFFERS
    // ============================================================
    if (size <= 0) {
        throw std::invalid_argument(
            "size debe ser mayor que cero"
        );
    }

    const std::size_t bytes =
        static_cast<std::size_t>(size) * sizeof(float);

    if (input_bo.size() < bytes || output_bo.size() < bytes) {
        throw std::invalid_argument(
            "Los buffers XRT no tienen capacidad suficiente"
        );
    }

    // ============================================================
    // 2. OBTENCIÓN DEL RUNTIME Y DEL KERNEL XRT
    // ============================================================
    auto& runtime = FpgaRuntime::instance();
    xrt::device& device = runtime.device();

    xrt::kernel kernel(
        device,
        runtime.uuid(),
        "relu6_forward"
    );

    // ============================================================
    // 3. USO DE BUFFERS EXTERNOS
    // ============================================================
    // Los buffers pertenecen al llamador (futuro gestor del runtime).
    // Aquí no se crean xrt::bo ni se sincronizan datos con la CPU.
    // El llamador prepara la entrada inicial y conserva los buffers
    // mientras sean necesarios para otras operaciones.

    // ============================================================
    // 4. EJECUCIÓN DEL KERNEL HLS
    // ============================================================
    auto run = kernel(
        input_bo,
        output_bo,
        size
    );

    // ============================================================
    // 5. FINALIZACIÓN DE LA EJECUCIÓN
    // ============================================================
    // Esperar a que termine no recupera la salida hacia la CPU.
    run.wait();

    // El resultado permanece en output_bo. El llamador decide si
    // encadena otro kernel o sincroniza el resultado final a CPU.
}

