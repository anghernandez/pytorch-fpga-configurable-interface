#include "fpga_runtime.hpp"
#include <iostream>

int main()
{
    auto& runtime = FpgaRuntime::instance();

    const std::size_t size = 1024;

    // Simulamos un grupo de memoria.
    // Primero usaremos el grupo que corresponda a uno de tus kernels.
    std::uint32_t group = 0;

    // Primera adquisición → debería CREAR el BO.
    auto bo1 = runtime.acquire_buffer(size, group);

    // Liberamos el BO.
    runtime.release_buffer(bo1);

    // Segunda adquisición → debería REUTILIZAR el mismo BO.
    auto bo2 = runtime.acquire_buffer(size, group);

    runtime.release_buffer(bo2);

    return 0;
}