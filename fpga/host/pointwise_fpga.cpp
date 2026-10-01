#include "pointwise_fpga.hpp"

#include "../../runtime/fpga_runtime.hpp"


#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>


void pointwise_fpga_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    bool use_bias
    //const std::string& xclbin_path
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
            "bias no puede ser nullptr cuando use_bias=true"
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

/*    if (xclbin_path.empty()) {
        throw std::invalid_argument(
            "xclbin_path no puede estar vacio"
        );
    }
*/


    const std::size_t input_elements =
        static_cast<std::size_t>(batch_size)
        * static_cast<std::size_t>(input_channels)
        * static_cast<std::size_t>(input_height)
        * static_cast<std::size_t>(input_width);

    const std::size_t weight_elements =
        static_cast<std::size_t>(output_channels)
        * static_cast<std::size_t>(input_channels);

    const std::size_t bias_elements =
        static_cast<std::size_t>(output_channels);

    const std::size_t output_elements =
        static_cast<std::size_t>(batch_size)
        * static_cast<std::size_t>(output_channels)
        * static_cast<std::size_t>(input_height)
        * static_cast<std::size_t>(input_width);


    const std::size_t input_bytes =
        input_elements * sizeof(float);

    const std::size_t weight_bytes =
        weight_elements * sizeof(float);

    const std::size_t bias_bytes =
        bias_elements * sizeof(float);

    const std::size_t output_bytes =
        output_elements * sizeof(float);


/*   xrt::device device(0);

    auto uuid =
        device.load_xclbin(xclbin_path);
*/

    auto& runtime = FpgaRuntime::instance();

    xrt::device& device = runtime.device();
    const xrt::uuid& uuid = runtime.uuid();

    xrt::kernel kernel(
        device,
        uuid,
        "pointwise_conv2d_forward"
    );


    xrt::bo input_bo(
        device,
        input_bytes,
        kernel.group_id(0)
    );

    xrt::bo weight_bo(
        device,
        weight_bytes,
        kernel.group_id(1)
    );

    xrt::bo bias_bo(
        device,
        bias_bytes,
        kernel.group_id(2)
    );

    xrt::bo output_bo(
        device,
        output_bytes,
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


    std::copy_n(
        input,
        input_elements,
        input_map
    );

    std::copy_n(
        weight,
        weight_elements,
        weight_map
    );


    if (use_bias) {
        std::copy_n(
            bias,
            bias_elements,
            bias_map
        );
    }
    else {
        std::fill_n(
            bias_map,
            bias_elements,
            0.0f
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
        use_bias
    );

    run.wait();


    output_bo.sync(
        XCL_BO_SYNC_BO_FROM_DEVICE
    );


    std::copy_n(
        output_map,
        output_elements,
        output
    );
}

