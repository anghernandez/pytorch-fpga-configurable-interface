#include "relu6_fpga.hpp"
#include "fpga_runtime.hpp"
#include "fpga_profile.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>

#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>

void relu6_fpga_forward(
    const float* input,
    float* output,
    int size
)
{
    if (input == nullptr) {
        throw std::invalid_argument("input no puede ser nullptr");
    }

    if (output == nullptr) {
        throw std::invalid_argument("output no puede ser nullptr");
    }

    if (size <= 0) {
        throw std::invalid_argument("size debe ser mayor que cero");
    }

    const std::size_t elements = static_cast<std::size_t>(size);
    const std::size_t bytes = elements * sizeof(float);

   
    auto& runtime = FpgaRuntime::instance();
    auto& device = runtime.device();
    const auto& uuid = runtime.uuid();


    {
        // 1. Creación del objeto kernel.
        xrt::kernel kernel(device, uuid, "relu6_forward");

        // 2. Consulta de grupos de memoria y reserva de buffers.
        xrt::bo input_bo(device, bytes, kernel.group_id(0));
        xrt::bo output_bo(device, bytes, kernel.group_id(1));

        // 3. Mapeo de buffers.
        float* input_map = input_bo.map<float*>();
        float* output_map = output_bo.map<float*>();

        // 4. Copia de entrada.
        std::copy_n(input, elements, input_map);

        // 5. Sincronización hacia FPGA.
        input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // 6. Lanzamiento.
        auto run = kernel(input_bo, output_bo, size);

        // 7. Espera.
        run.wait();

        // 8. Sincronización hacia CPU.
        output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        // 9. Copia de salida.
        std::copy_n(output_map, elements, output);

    }

  
}