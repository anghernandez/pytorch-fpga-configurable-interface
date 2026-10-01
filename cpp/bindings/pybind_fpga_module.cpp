#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "tanh_fpga.hpp"


namespace py = pybind11;


using FloatArray = py::array_t<
    float,
    py::array::c_style
>;


py::array_t<float> tanh_fpga_forward_binding(
    const FloatArray& input
)
{
    const py::buffer_info input_info =
        input.request();

    if (input_info.size <= 0) {
        throw py::value_error(
            "La entrada no puede estar vacía"
        );
    }

    py::array_t<float> output(
        input_info.shape
    );

    py::buffer_info output_info =
        output.request();

    const auto* input_pointer =
        static_cast<const float*>(
            input_info.ptr
        );

    auto* output_pointer =
        static_cast<float*>(
            output_info.ptr
        );

    tanh_fpga_forward(
        input_pointer,
        output_pointer,
        static_cast<int>(input_info.size)
    );

    return output;
}


PYBIND11_MODULE(fpga_kernels, module)
{
    module.doc() =
        "Kernels HLS ejecutados en FPGA mediante XRT";

    module.def(
        "tanh_forward",
        &tanh_fpga_forward_binding,
        py::arg("input"),
        "Ejecuta Tanh en la FPGA sobre un arreglo NumPy float32"
    );
}
