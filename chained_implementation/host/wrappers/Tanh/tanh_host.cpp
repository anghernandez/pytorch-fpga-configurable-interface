#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>
#include <xrt/xrt/xrt_bo.h>

int main()
{
    const int size = 301056;
    const size_t bytes =
        static_cast<size_t>(size) * sizeof(float);

    std::vector<float> input(size);
    std::vector<float> output(size, 0.0f);

    for (int i = 0; i < size; i++)
    {
        input[i] =
            static_cast<float>((i % 41) - 20) * 0.1f;
    }

    std::cout << "Abriendo dispositivo...\n";

    xrt::device device(0);

    std::cout << "Cargando xclbin...\n";

    auto uuid = device.load_xclbin(
        "/lib/firmware/xilinx/tanh/tanh.xclbin"
    );

    std::cout << "Abriendo kernel tanh_forward...\n";

    xrt::kernel kernel(
        device,
        uuid,
        "tanh_forward"
    );

    std::cout << "Creando buffers...\n";

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

    float* input_map =
        input_bo.map<float*>();

    float* output_map =
        output_bo.map<float*>();

    for (int i = 0; i < size; i++)
    {
        input_map[i] = input[i];
        output_map[i] = 0.0f;
    }

    std::cout << "Enviando datos a FPGA...\n";

    input_bo.sync(
        XCL_BO_SYNC_BO_TO_DEVICE
    );

    std::cout << "Ejecutando kernel...\n";

    auto run = kernel(
        input_bo,
        output_bo,
        size
    );

    run.wait();

    std::cout << "Leyendo resultados...\n";

    output_bo.sync(
        XCL_BO_SYNC_BO_FROM_DEVICE
    );

    double squared_error_sum = 0.0;
    float maximum_error = 0.0f;

    for (int i = 0; i < size; i++)
    {
        const float hw =
            output_map[i];

        const float sw =
            std::tanh(input[i]);

        const float error =
            std::fabs(hw - sw);

        squared_error_sum +=
            static_cast<double>(error)
            * static_cast<double>(error);

        maximum_error =
            std::max(maximum_error, error);

        output[i] = hw;
    }

    const double rmse =
        std::sqrt(
            squared_error_sum /
            static_cast<double>(size)
        );

    const bool passed =
        maximum_error <= 1.0e-5f;

    std::cout << std::scientific
              << std::setprecision(8);

    std::cout
        << "\nSize      = " << size
        << "\nRMSE      = " << rmse
        << "\nMax error = " << maximum_error
        << "\nResult    = "
        << (passed ? "PASS" : "FAIL")
        << "\n";

    return passed ? 0 : 1;
}