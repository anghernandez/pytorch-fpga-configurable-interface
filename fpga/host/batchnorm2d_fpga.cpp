#include "batchnorm2d_fpga.hpp"
#include "../../runtime/fpga_runtime.hpp"


#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>


void batchnorm2d_fpga_forward(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    float eps,
    bool affine,
    bool use_bias
   // const std::string& xclbin_path
)
{
    if (input == nullptr) {
        throw std::invalid_argument(
            "input no puede ser nullptr"
        );
    }

    if (running_mean == nullptr) {
        throw std::invalid_argument(
            "running_mean no puede ser nullptr"
        );
    }

    if (running_var == nullptr) {
        throw std::invalid_argument(
            "running_var no puede ser nullptr"
        );
    }

    if (output == nullptr) {
        throw std::invalid_argument(
            "output no puede ser nullptr"
        );
    }

    if (affine && weight == nullptr) {
        throw std::invalid_argument(
            "weight no puede ser nullptr cuando affine=true"
        );
    }

    if (affine && use_bias && bias == nullptr) {
        throw std::invalid_argument(
            "bias no puede ser nullptr cuando use_bias=true"
        );
    }

    if (
        batch_size <= 0 ||
        channels <= 0 ||
        input_height <= 0 ||
        input_width <= 0
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

    const std::size_t element_count =
        static_cast<std::size_t>(batch_size)
        * static_cast<std::size_t>(channels)
        * static_cast<std::size_t>(input_height)
        * static_cast<std::size_t>(input_width);

    const std::size_t input_bytes =
        element_count * sizeof(float);

    const std::size_t parameter_bytes =
        static_cast<std::size_t>(channels)
        * sizeof(float);


/*    xrt::device device(0);

    auto uuid =
        device.load_xclbin(xclbin_path);*/

    auto& runtime = FpgaRuntime::instance();

    xrt::device& device = runtime.device();
    const xrt::uuid& uuid = runtime.uuid();

    xrt::kernel kernel(
        device,
        uuid,
        "batchnorm2d_forward"
    );


    xrt::bo input_bo(
        device,
        input_bytes,
        kernel.group_id(0)
    );

    xrt::bo weight_bo(
        device,
        parameter_bytes,
        kernel.group_id(1)
    );

    xrt::bo bias_bo(
        device,
        parameter_bytes,
        kernel.group_id(2)
    );

    xrt::bo running_mean_bo(
        device,
        parameter_bytes,
        kernel.group_id(3)
    );

    xrt::bo running_var_bo(
        device,
        parameter_bytes,
        kernel.group_id(4)
    );

    xrt::bo output_bo(
        device,
        input_bytes,
        kernel.group_id(5)
    );


    float* input_map =
        input_bo.map<float*>();

    float* weight_map =
        weight_bo.map<float*>();

    float* bias_map =
        bias_bo.map<float*>();

    float* running_mean_map =
        running_mean_bo.map<float*>();

    float* running_var_map =
        running_var_bo.map<float*>();

    float* output_map =
        output_bo.map<float*>();


    std::copy_n(
        input,
        element_count,
        input_map
    );

    std::copy_n(
        running_mean,
        channels,
        running_mean_map
    );

    std::copy_n(
        running_var,
        channels,
        running_var_map
    );


    if (affine) {
        std::copy_n(
            weight,
            channels,
            weight_map
        );
    }
    else {
        std::fill_n(
            weight_map,
            channels,
            1.0f
        );
    }


    if (affine && use_bias) {
        std::copy_n(
            bias,
            channels,
            bias_map
        );
    }
    else {
        std::fill_n(
            bias_map,
            channels,
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

    running_mean_bo.sync(
        XCL_BO_SYNC_BO_TO_DEVICE
    );

    running_var_bo.sync(
        XCL_BO_SYNC_BO_TO_DEVICE
    );


    auto run = kernel(
        input_bo,
        weight_bo,
        bias_bo,
        running_mean_bo,
        running_var_bo,
        output_bo,
        batch_size,
        channels,
        input_height,
        input_width,
        eps,
        affine,
        use_bias
    );

    run.wait();


    output_bo.sync(
        XCL_BO_SYNC_BO_FROM_DEVICE
    );


    std::copy_n(
        output_map,
        element_count,
        output
    );
}