#include "../ReLU6/SW/relu6_fpga.hpp"
#include "../runtime/fpga_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <xrt/xrt/xrt_kernel.h>

int main()
{
    constexpr int size = 8;

    // Valores que permiten comprobar:
    // negativos, valores entre 0 y 6
    // y valores mayores que 6.
    const std::vector<float> input = {
        -10.0f,
        -2.0f,
         0.0f,
         2.0f,
         5.0f,
         6.0f,
         8.0f,
        20.0f
    };

    const std::vector<float> expected = {
        0.0f,
        0.0f,
        0.0f,
        2.0f,
        5.0f,
        6.0f,
        6.0f,
        6.0f
    };

    auto& runtime = FpgaRuntime::instance();

    // Creamos el kernel para obtener sus grupos de memoria.
    xrt::kernel kernel(
        runtime.device(),
        runtime.uuid(),
        "relu6_forward"
    );

    const std::uint32_t input_group =
        kernel.group_id(0);

    const std::uint32_t output_group =
        kernel.group_id(1);

    // Reservamos los buffers desde el runtime.
    FpgaBuffer input_buffer =
        runtime.acquire_buffer(
            size * sizeof(float),
            input_group
        );

    FpgaBuffer output_buffer =
        runtime.acquire_buffer(
            size * sizeof(float),
            output_group
        );

    // Mapeamos el buffer de entrada.
    float* input_map =
        input_buffer.bo.map<float*>();

    // Copiamos los datos de prueba al BO.
    std::copy(
        input.begin(),
        input.end(),
        input_map
    );

    // Enviamos el buffer de entrada a la FPGA.
    input_buffer.bo.sync(
        XCL_BO_SYNC_BO_TO_DEVICE
    );

    // Ejecutamos ReLU6 usando directamente los FpgaBuffer.
    relu6_fpga_forward_buffer(
        input_buffer,
        output_buffer,
        size
    );

    // Solo hacemos esta transferencia porque necesitamos
    // recuperar el resultado para verificarlo.
    output_buffer.bo.sync(
        XCL_BO_SYNC_BO_FROM_DEVICE
    );

    float* output_map =
        output_buffer.bo.map<float*>();

    bool passed = true;

    for (int i = 0; i < size; ++i)
    {
        const float error =
            std::abs(output_map[i] - expected[i]);

        std::cout
            << "input=" << input[i]
            << " | FPGA=" << output_map[i]
            << " | esperado=" << expected[i]
            << " | error=" << error
            << std::endl;

        if (error > 1e-5f)
        {
            passed = false;
        }
    }

    runtime.release_buffer(input_buffer);
    runtime.release_buffer(output_buffer);

    std::cout << "\nRESULTADO: "
              << (passed ? "PASS" : "FAIL")
              << std::endl;

    return passed ? 0 : 1;
}