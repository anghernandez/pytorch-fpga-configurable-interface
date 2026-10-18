#include "tanh_fpga.hpp"
#include "fpga_runtime.hpp"

#include <cstddef>
#include <stdexcept>

#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>



void tanh_fpga_forward(
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
    // El runtime compartido ya gestiona el dispositivo y el XCLBIN.
    auto& runtime = FpgaRuntime::instance();
    xrt::device& device = runtime.device();

    xrt::kernel kernel(
        device,
        runtime.uuid(),
        "tanh_forward"
    );

    // ============================================================
    // 3. USO DE BUFFERS EXTERNOS
    // ============================================================
    // input_bo y output_bo pertenecen al llamador (futuro runtime).
    // No se crean xrt::bo locales ni se copian datos desde la CPU.
    // El llamador debe preparar/sincronizar la entrada inicial
    // cuando provenga de memoria del host.

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
    // Esperar no implica copiar la salida a la CPU.
    run.wait();

    // El resultado permanece en output_bo. El llamador decide
    // qué kernel lo consume después o cuándo recuperarlo a CPU.
}