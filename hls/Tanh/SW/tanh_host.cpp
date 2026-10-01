#include <iostream>
#include <vector>
#include <cmath>

#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>
#include <xrt/xrt/xrt_bo.h>

int main()
{
    const int size = 5;
    const size_t bytes = size * sizeof(float);

    std::vector<float> input = {
        -2.0f,
        -1.0f,
         0.0f,
         1.0f,
         2.0f
    };

    std::vector<float> output(size, 0.0f);

    std::cout << "Abriendo dispositivo..." << std::endl;

    xrt::device device(0);

    std::cout << "Cargando xclbin..." << std::endl;

    auto uuid = device.load_xclbin(
        "/lib/firmware/xilinx/tanh/tanh.xclbin"
    );

    std::cout << "Abriendo kernel tanh_forward..." << std::endl;

    xrt::kernel kernel(
        device,
        uuid,
        "tanh_forward"
    );

    std::cout << "Creando buffers..." << std::endl;

    xrt::bo input_bo(
        device,
        bytes,
        kernel.group_id(0)
    );

    xrt::bo output_bo(
        device,
        bytes,
        kernel.group_id(1)
    );

    float* input_map = input_bo.map<float*>();
    float* output_map = output_bo.map<float*>();

    for (int i = 0; i < size; i++)
    {
        input_map[i] = input[i];
        output_map[i] = 0.0f;
    }

    std::cout << "Enviando datos a FPGA..." << std::endl;

    input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "Ejecutando kernel..." << std::endl;

    auto run = kernel(
        input_bo,
        output_bo,
        size
    );

    run.wait();

    std::cout << "Leyendo resultados..." << std::endl;

    output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    bool passed = true;

    std::cout << "\nResultados:\n";

    for (int i = 0; i < size; i++)
    {
        float hw = output_map[i];
        float sw = std::tanh(input[i]);
        float error = std::fabs(hw - sw);

        std::cout
            << "x = " << input[i]
            << " | HW = " << hw
            << " | SW = " << sw
            << " | error = " << error
            << std::endl;

        if (error > 1e-5f)
        {
            passed = false;
        }
    }

    if (passed)
    {
        std::cout << "\nTEST PASSED" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "\nTEST FAILED" << std::endl;
        return 1;
    }
}
