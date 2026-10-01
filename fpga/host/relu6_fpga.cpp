#include "relu6_fpga.hpp"

#include "../../runtime/fpga_runtime.hpp"


#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>


void relu6_fpga_forward(
    const float* input,
    float* output,
    int size
   // const std::string& xclbin_path
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

/*    if (xclbin_path.empty()) {
        throw std::invalid_argument(
            "xclbin_path no puede estar vacio"
        );
    }
*/

    const std::size_t elements =
        static_cast<std::size_t>(size);

    const std::size_t bytes =
        elements * sizeof(float);


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
        "relu6_forward"
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
        elements,
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
        elements,
        output
    );
}
