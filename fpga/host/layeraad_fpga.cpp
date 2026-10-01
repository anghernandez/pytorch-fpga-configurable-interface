#include "layeradd_fpga.hpp"

#include "../../runtime/fpga_runtime.hpp"


#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>


void layeradd_fpga_forward(
    const float* input1,
    const float* input2,
    float* output,
    int size,
    float alpha

)
{
    if (input1 == nullptr) {
        throw std::invalid_argument(
            "input1 no puede ser nullptr"
        );
    }

    if (input2 == nullptr) {
        throw std::invalid_argument(
            "input2 no puede ser nullptr"
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

/*    if (xclbin_path.empty()) {
        throw std::invalid_argument(
            "xclbin_path no puede estar vacio"
        );
    }*/


    const std::size_t element_count =
        static_cast<std::size_t>(size);

    const std::size_t buffer_bytes =
        element_count * sizeof(float);


/*    xrt::device device(0);

    auto uuid =
        device.load_xclbin(xclbin_path);
*/

    auto& runtime = FpgaRuntime::instance();

    xrt::device& device = runtime.device();
    const xrt::uuid& uuid = runtime.uuid();

    xrt::kernel kernel(
        device,
        uuid,
        "layer_add_forward"
    );


    xrt::bo input1_bo(
        device,
        buffer_bytes,
        kernel.group_id(0)
    );

    xrt::bo input2_bo(
        device,
        buffer_bytes,
        kernel.group_id(1)
    );

    xrt::bo output_bo(
        device,
        buffer_bytes,
        kernel.group_id(2)
    );


    float* input1_map =
        input1_bo.map<float*>();

    float* input2_map =
        input2_bo.map<float*>();

    float* output_map =
        output_bo.map<float*>();


    std::copy_n(
        input1,
        element_count,
        input1_map
    );

    std::copy_n(
        input2,
        element_count,
        input2_map
    );


    input1_bo.sync(
        XCL_BO_SYNC_BO_TO_DEVICE
    );

    input2_bo.sync(
        XCL_BO_SYNC_BO_TO_DEVICE
    );


    auto run = kernel(
        input1_bo,
        input2_bo,
        output_bo,
        size,
        alpha
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