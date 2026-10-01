#include "linear_fpga.hpp"
// Permite el runtime
#include "../../runtime/fpga_runtime.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>


void linear_fpga_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
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
        in_features <= 0 ||
        out_features <= 0
    ) {
        throw std::invalid_argument(
            "Las dimensiones deben ser mayores que cero"
        );
    }

    const std::size_t input_size =
        static_cast<std::size_t>(batch_size)
        * static_cast<std::size_t>(in_features);

    const std::size_t weight_size =
        static_cast<std::size_t>(out_features)
        * static_cast<std::size_t>(in_features);

    const std::size_t bias_size =
        static_cast<std::size_t>(out_features);

    const std::size_t output_size =
        static_cast<std::size_t>(batch_size)
        * static_cast<std::size_t>(out_features);


    if (batch_size > 64) {
        throw std::invalid_argument(
            "batch_size excede el maximo soportado de 64"
        );
    }


    if (in_features > 1280) {
        throw std::invalid_argument(
            "in_features excede el maximo soportado de 1280"
        );
    }

    if (out_features > 1000) {
        throw std::invalid_argument(
            "out_features excede el maximo soportado de 1000"
        );
    }

            
    if (input_size > 81920) {
        throw std::invalid_argument(
            "input excede el depth HLS de 81920 elementos"
        );
    }

    if (weight_size > 1280000) {
        throw std::invalid_argument(
            "weight excede el depth HLS de 1280000 elementos"
        );
    }

    if (bias_size > 1000) {
        throw std::invalid_argument(
            "bias excede el depth HLS de 1000 elementos"
        );
    }

    if (output_size > 64000) {
        throw std::invalid_argument(
            "output excede el depth HLS de 64000 elementos"
        );
    }

    const std::size_t input_bytes =
        input_size * sizeof(float);

    const std::size_t weight_bytes =
        weight_size * sizeof(float);

    const std::size_t bias_bytes =
        bias_size * sizeof(float);

    const std::size_t output_bytes =
        output_size * sizeof(float);


    /* Implementación anterior sin runtime    
    const std::string binary_file =
        "/lib/firmware/xilinx/linear/linear.xclbin";

    xrt::device device(0);

    auto uuid = device.load_xclbin(
        binary_file
    );

    xrt::kernel kernel(
        device,
        uuid,
        "linear_forward"
    );
    */

    // Implementación con runtime
    auto& runtime = FpgaRuntime::instance();

    xrt::device& device = runtime.device();

    xrt::kernel kernel(
        device,
        runtime.uuid(),
        "linear_forward"
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
    } else {
        std::fill_n(
            bias_map,
            bias_size,
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
        in_features,
        out_features,
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