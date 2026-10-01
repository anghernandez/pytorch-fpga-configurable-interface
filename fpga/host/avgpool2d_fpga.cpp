#include "avgpool2d_fpga.hpp"
// Permite el runtime
#include "../../runtime/fpga_runtime.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>


void avgpool2d_fpga_forward(
    const float* input,
    float* output,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool count_include_pad,
    bool use_divisor_override,
    int divisor_override
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

    if (
        batch_size <= 0 ||
        input_channels <= 0 ||
        input_height <= 0 ||
        input_width <= 0
    ) {
        throw std::invalid_argument(
            "Las dimensiones de entrada deben ser mayores que cero"
        );
    }

    if (
        output_height <= 0 ||
        output_width <= 0
    ) {
        throw std::invalid_argument(
            "Las dimensiones de salida deben ser mayores que cero"
        );
    }

    if (
        kernel_height <= 0 ||
        kernel_width <= 0
    ) {
        throw std::invalid_argument(
            "El tamaño del kernel debe ser mayor que cero"
        );
    }

    if (
        stride_height <= 0 ||
        stride_width <= 0
    ) {
        throw std::invalid_argument(
            "El stride debe ser mayor que cero"
        );
    }

    if (
        padding_height < 0 ||
        padding_width < 0
    ) {
        throw std::invalid_argument(
            "El padding no puede ser negativo"
        );
    }

    if (
        use_divisor_override &&
        divisor_override <= 0
    ) {
        throw std::invalid_argument(
            "divisor_override debe ser mayor que cero"
        );
    }

    const int expected_output_height =
        (
            input_height
            + 2 * padding_height
            - kernel_height
        ) / stride_height
        + 1;

    const int expected_output_width =
        (
            input_width
            + 2 * padding_width
            - kernel_width
        ) / stride_width
        + 1;

    if (
        output_height != expected_output_height ||
        output_width != expected_output_width
    ) {
        throw std::invalid_argument(
            "Las dimensiones de salida no coinciden con "
            "input, kernel, stride y padding"
        );
    }

    const std::size_t input_size =
        static_cast<std::size_t>(batch_size)
        * static_cast<std::size_t>(input_channels)
        * static_cast<std::size_t>(input_height)
        * static_cast<std::size_t>(input_width);

    const std::size_t output_size =
        static_cast<std::size_t>(batch_size)
        * static_cast<std::size_t>(input_channels)
        * static_cast<std::size_t>(output_height)
        * static_cast<std::size_t>(output_width);



    const std::size_t input_bytes =
        input_size * sizeof(float);

    const std::size_t output_bytes =
        output_size * sizeof(float);

    /*
    const std::string binary_file =
        "/lib/firmware/xilinx/avgpool2d/avgpool2d.xclbin";

    xrt::device device(0);

    auto uuid = device.load_xclbin(
        binary_file
    );

    xrt::kernel kernel(
        device,
        uuid,
        "avg_pool2d_forward"
    );
    */
    // Implementación con runtime   
    auto& runtime = FpgaRuntime::instance();

    xrt::device& device = runtime.device();

    xrt::kernel kernel(
        device,
        runtime.uuid(),
        "avg_pool2d_forward"
    );


    xrt::bo input_bo(
        device,
        input_bytes,
        kernel.group_id(0)
    );

    xrt::bo output_bo(
        device,
        output_bytes,
        kernel.group_id(1)
    );

    float* input_map =
        input_bo.map<float*>();

    float* output_map =
        output_bo.map<float*>();

    std::copy_n(
        input,
        input_size,
        input_map
    );

    input_bo.sync(
        XCL_BO_SYNC_BO_TO_DEVICE
    );

    auto run = kernel(
        input_bo,
        output_bo,
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        count_include_pad,
        use_divisor_override,
        divisor_override
    );

    run.wait();

    output_bo.sync(
        XCL_BO_SYNC_BO_FROM_DEVICE
    );

    std::copy_n(
        output_map,
        output_size,
        output
    );
}

