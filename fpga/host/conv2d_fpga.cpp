#include "conv2d_fpga.hpp"
// Permite el runtime
#include "../../runtime/fpga_runtime.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
//#include <string>

#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>


void conv2d_fpga_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool use_bias
)
{
    if (input == nullptr) {
        throw std::invalid_argument(
            "input no puede ser nullptr"
        );
    }

    if (weight == nullptr) {
        throw std::invalid_argument(
            "weight no puede ser nullptr"
        );
    }

    if (output == nullptr) {
        throw std::invalid_argument(
            "output no puede ser nullptr"
        );
    }

    if (use_bias && bias == nullptr) {
        throw std::invalid_argument(
            "bias no puede ser nullptr cuando use_bias es true"
        );
    }

    if (
        batch_size <= 0 ||
        input_channels <= 0 ||
        input_height <= 0 ||
        input_width <= 0 ||
        output_channels <= 0
    ) {
        throw std::invalid_argument(
            "Las dimensiones deben ser mayores que cero"
        );
    }

    

    if (
        kernel_height <= 0 ||
        kernel_width <= 0
    ) {
        throw std::invalid_argument(
            "El kernel debe ser mayor que cero"
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
            "Las dimensiones de salida son incorrectas"
        );
    }

    if (
        output_height <= 0 ||
        output_width <= 0
    ) {
        throw std::invalid_argument(
            "La configuración produce una salida vacía"
        );
    }

    const std::size_t input_size =
        static_cast<std::size_t>(batch_size)
        * static_cast<std::size_t>(input_channels)
        * static_cast<std::size_t>(input_height)
        * static_cast<std::size_t>(input_width);

    const std::size_t weight_size =
        static_cast<std::size_t>(output_channels)
        * static_cast<std::size_t>(input_channels)
        * static_cast<std::size_t>(kernel_height)
        * static_cast<std::size_t>(kernel_width);

    const std::size_t bias_size =
        static_cast<std::size_t>(output_channels);

    const std::size_t output_size =
        static_cast<std::size_t>(batch_size)
        * static_cast<std::size_t>(output_channels)
        * static_cast<std::size_t>(output_height)
        * static_cast<std::size_t>(output_width);


  

/* Implementación anterior sin runtime
    const std::string binary_file =
        "/lib/firmware/xilinx/conv2d/conv2d.xclbin";

    xrt::device device(0);

    auto uuid =
        device.load_xclbin(binary_file);

    xrt::kernel kernel(
        device,
        uuid,
        "conv2d_forward"
    );
*/

    // Implementación con runtime
    auto& runtime = FpgaRuntime::instance();

    xrt::device& device = runtime.device();

    xrt::kernel kernel(
        device,
        runtime.uuid(),
        "conv2d_forward"
    );

    xrt::bo input_bo(
        device,
        input_size * sizeof(float),
        kernel.group_id(0)
    );

    xrt::bo weight_bo(
        device,
        weight_size * sizeof(float),
        kernel.group_id(1)
    );

    xrt::bo bias_bo(
        device,
        bias_size * sizeof(float),
        kernel.group_id(2)
    );

    xrt::bo output_bo(
        device,
        output_size * sizeof(float),
        kernel.group_id(3)
    );

    float* input_map =
        input_bo.map<float*>();

    float* weight_map =
        weight_bo.map<float*>();

    float* bias_map =
        bias_bo.map<float*>();

    float* output_map =
        output_bo.map<float*>();

    std::fill_n(
        input_map,
        input_size,
        0.0f
    );

    std::fill_n(
        weight_map,
        weight_size,
        0.0f
    );

    std::fill_n(
        bias_map,
        bias_size,
        0.0f
    );

    std::fill_n(
        output_map,
        output_size,
        0.0f
    );

    std::copy_n(
        input,
        input_size,
        input_map
    );

    std::copy_n(
        weight,
        weight_size,
        weight_map
    );

    if (use_bias) {
        std::copy_n(
            bias,
            bias_size,
            bias_map
        );
    }

    input_bo.sync(
        XCL_BO_SYNC_BO_TO_DEVICE
    );

    weight_bo.sync(
        XCL_BO_SYNC_BO_TO_DEVICE
    );

    bias_bo.sync(
        XCL_BO_SYNC_BO_TO_DEVICE
    );

    output_bo.sync(
        XCL_BO_SYNC_BO_TO_DEVICE
    );

    auto run = kernel(
        input_bo,
        weight_bo,
        bias_bo,
        output_bo,
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_channels,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        use_bias
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
