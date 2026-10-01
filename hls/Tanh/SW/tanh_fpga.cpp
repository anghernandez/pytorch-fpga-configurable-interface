#include "tanh_fpga.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>


void tanh_fpga_forward(
    const float* input,
    float* output,
    int size
)
{
    if (input == nullptr) {
        throw std::invalid_argument(
            "input no puede ser nullptr"
        );
    }

    if (output == nullptr) {
        throw std::invalid_argument(
            "output no puede ser nullptr"
        );
    }

    if (size <= 0) {
        throw std::invalid_argument(
            "size debe ser mayor que cero"
        );
    }

    const std::size_t bytes =
        static_cast<std::size_t>(size)
        * sizeof(float);

    const std::string binary_file =
        "/lib/firmware/xilinx/tanh/tanh.xclbin";

    xrt::device device(0);

    auto uuid = device.load_xclbin(
        binary_file
    );

    xrt::kernel kernel(
        device,
        uuid,
        "tanh_forward"
    );

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

    std::copy_n(
        input,
        size,
        input_map
    );

    input_bo.sync(
        XCL_BO_SYNC_BO_TO_DEVICE
    );

    auto run = kernel(
        input_bo,
        output_bo,
        size
    );

    run.wait();

    output_bo.sync(
        XCL_BO_SYNC_BO_FROM_DEVICE
    );

    std::copy_n(
        output_map,
        size,
        output
    );
}

