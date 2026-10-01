#include <iostream>
#include "fpga_runtime.hpp"

int main()
{
    std::cout << "Antes de obtener runtime" << std::endl;

    auto& runtime = FpgaRuntime::instance();

    std::cout << "Runtime creado correctamente" << std::endl;

    auto& device = runtime.device();

    std::cout << "Device obtenido correctamente" << std::endl;

    return 0;
}
