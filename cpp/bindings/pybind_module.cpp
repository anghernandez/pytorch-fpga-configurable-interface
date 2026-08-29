#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ReLU.hpp"
#include "Tanh.hpp"
#include "Linear.hpp"
#include "AvgPool2d.hpp"
#include "Conv2d.hpp"

namespace py = pybind11;


//Alias para arreglos continuos
using FloatArray = py::array_t<
    float,
    py::array::c_style
>;


//ReLU
py::array_t<float> relu_forward_binding(
    const py::array_t<
        float,
        py::array::c_style
    >& input
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

    relu_forward(
        input_pointer,
        output_pointer,
        static_cast<int>(input_info.size)
    );

    return output;
}

//Tanh

py::array_t<float> tanh_forward_binding(
    const py::array_t<
        float,
        py::array::c_style
    >& input
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

    tanh_forward(
        input_pointer,
        output_pointer,
        static_cast<int>(input_info.size)
    );

    return output;
}

//Linear

py::array_t<float> linear_forward_binding(
    const FloatArray& input,
    const FloatArray& weight,
    const py::object& bias
)
{
    const py::buffer_info input_info =
        input.request();

    const py::buffer_info weight_info =
        weight.request();

    if (input_info.ndim != 2) {
        throw py::value_error(
            "input debe tener forma "
            "[batch_size, in_features]"
        );
    }

    if (weight_info.ndim != 2) {
        throw py::value_error(
            "weight debe tener forma "
            "[out_features, in_features]"
        );
    }

    const py::ssize_t batch_size =
        input_info.shape[0];

    const py::ssize_t in_features =
        input_info.shape[1];

    const py::ssize_t out_features =
        weight_info.shape[0];

    if (weight_info.shape[1] != in_features) {
        throw py::value_error(
            "input y weight tienen "
            "in_features incompatibles"
        );
    }

    py::array_t<float> output(
        {
            batch_size,
            out_features
        }
    );

    py::buffer_info output_info =
        output.request();

    const auto* input_pointer =
        static_cast<const float*>(
            input_info.ptr
        );

    const auto* weight_pointer =
        static_cast<const float*>(
            weight_info.ptr
        );

    auto* output_pointer =
        static_cast<float*>(
            output_info.ptr
        );

    if (bias.is_none()) {
        linear_forward(
            input_pointer,
            weight_pointer,
            nullptr,
            output_pointer,
            static_cast<int>(batch_size),
            static_cast<int>(in_features),
            static_cast<int>(out_features),
            false
        );

        return output;
    }

    FloatArray bias_array =
        FloatArray::ensure(bias);

    if (!bias_array) {
        throw py::type_error(
            "bias debe ser un arreglo NumPy "
            "float32 y contiguo, o None"
        );
    }

    const py::buffer_info bias_info =
        bias_array.request();

    if (
        bias_info.ndim != 1
        || bias_info.shape[0] != out_features
    ) {
        throw py::value_error(
            "bias debe tener forma [out_features]"
        );
    }

    const auto* bias_pointer =
        static_cast<const float*>(
            bias_info.ptr
        );

    linear_forward(
        input_pointer,
        weight_pointer,
        bias_pointer,
        output_pointer,
        static_cast<int>(batch_size),
        static_cast<int>(in_features),
        static_cast<int>(out_features),
        true
    );

    return output;
}


//AvgPool2d

py::array_t<float> avg_pool2d_forward_binding(
    const FloatArray& input,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool count_include_pad,
    const py::object& divisor_override
)
{
    const py::buffer_info input_info =
        input.request();

    if (input_info.ndim != 4) {
        throw py::value_error(
            "input debe tener forma "
            "[batch, channels, height, width]"
        );
    }

    if (
        output_height <= 0
        || output_width <= 0
    ) {
        throw py::value_error(
            "Las dimensiones de salida deben "
            "ser mayores que cero"
        );
    }

    const py::ssize_t batch_size =
        input_info.shape[0];

    const py::ssize_t input_channels =
        input_info.shape[1];

    const py::ssize_t input_height =
        input_info.shape[2];

    const py::ssize_t input_width =
        input_info.shape[3];

    py::array_t<float> output(
        {
            batch_size,
            input_channels,
            static_cast<py::ssize_t>(
                output_height
            ),
            static_cast<py::ssize_t>(
                output_width
            )
        }
    );

    py::buffer_info output_info =
        output.request();

    bool use_divisor_override = false;
    int divisor_override_value = 0;

    if (!divisor_override.is_none()) {
        if (
            !py::isinstance<py::int_>(
                divisor_override
            )
            || py::isinstance<py::bool_>(
                divisor_override
            )
        ) {
            throw py::type_error(
                "divisor_override debe ser "
                "un entero o None"
            );
        }

        divisor_override_value =
            divisor_override.cast<int>();

        if (divisor_override_value == 0) {
            throw py::value_error(
                "divisor_override no puede ser 0"
            );
        }

        use_divisor_override = true;
    }

    const auto* input_pointer =
        static_cast<const float*>(
            input_info.ptr
        );

    auto* output_pointer =
        static_cast<float*>(
            output_info.ptr
        );

    avg_pool2d_forward(
        input_pointer,
        output_pointer,
        static_cast<int>(batch_size),
        static_cast<int>(input_channels),
        static_cast<int>(input_height),
        static_cast<int>(input_width),
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
        divisor_override_value
    );

    return output;
}

//Conv2d
py::array_t<float> conv2d_forward_binding(
    const FloatArray& input,
    const FloatArray& weight,
    const py::object& bias,
    int output_height,
    int output_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width
)
{
    const py::buffer_info input_info =
        input.request();

    const py::buffer_info weight_info =
        weight.request();

    if (input_info.ndim != 4) {
        throw py::value_error(
            "input debe tener forma "
            "[batch, in_channels, height, width]"
        );
    }

    if (weight_info.ndim != 4) {
        throw py::value_error(
            "weight debe tener forma "
            "[out_channels, in_channels, "
            "kernel_height, kernel_width]"
        );
    }

    const py::ssize_t batch_size =
        input_info.shape[0];

    const py::ssize_t input_channels =
        input_info.shape[1];

    const py::ssize_t input_height =
        input_info.shape[2];

    const py::ssize_t input_width =
        input_info.shape[3];

    const py::ssize_t output_channels =
        weight_info.shape[0];

    const py::ssize_t weight_input_channels =
        weight_info.shape[1];

    const py::ssize_t kernel_height =
        weight_info.shape[2];

    const py::ssize_t kernel_width =
        weight_info.shape[3];

    if (
        input_channels
        != weight_input_channels
    ) {
        throw py::value_error(
            "Los canales de input y weight "
            "no son compatibles"
        );
    }

    if (
        output_height <= 0
        || output_width <= 0
    ) {
        throw py::value_error(
            "Las dimensiones de salida deben "
            "ser mayores que cero"
        );
    }

    py::array_t<float> output(
        {
            batch_size,
            output_channels,
            static_cast<py::ssize_t>(
                output_height
            ),
            static_cast<py::ssize_t>(
                output_width
            )
        }
    );

    py::buffer_info output_info =
        output.request();

    const auto* input_pointer =
        static_cast<const float*>(
            input_info.ptr
        );

    const auto* weight_pointer =
        static_cast<const float*>(
            weight_info.ptr
        );

    auto* output_pointer =
        static_cast<float*>(
            output_info.ptr
        );

    if (bias.is_none()) {
        conv2d_forward(
            input_pointer,
            weight_pointer,
            nullptr,
            output_pointer,
            static_cast<int>(batch_size),
            static_cast<int>(input_channels),
            static_cast<int>(input_height),
            static_cast<int>(input_width),
            static_cast<int>(output_channels),
            output_height,
            output_width,
            static_cast<int>(kernel_height),
            static_cast<int>(kernel_width),
            stride_height,
            stride_width,
            padding_height,
            padding_width,
            false
        );

        return output;
    }

    FloatArray bias_array =
        FloatArray::ensure(bias);

    if (!bias_array) {
        throw py::type_error(
            "bias debe ser un arreglo NumPy "
            "float32 y contiguo, o None"
        );
    }

    const py::buffer_info bias_info =
        bias_array.request();

    if (
        bias_info.ndim != 1
        || bias_info.shape[0]
            != output_channels
    ) {
        throw py::value_error(
            "bias debe tener forma [out_channels]"
        );
    }

    const auto* bias_pointer =
        static_cast<const float*>(
            bias_info.ptr
        );

    conv2d_forward(
        input_pointer,
        weight_pointer,
        bias_pointer,
        output_pointer,
        static_cast<int>(batch_size),
        static_cast<int>(input_channels),
        static_cast<int>(input_height),
        static_cast<int>(input_width),
        static_cast<int>(output_channels),
        output_height,
        output_width,
        static_cast<int>(kernel_height),
        static_cast<int>(kernel_width),
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        true
    );

    return output;
}

PYBIND11_MODULE(cpp_kernels, module)
{
    module.doc() =
        "Kernels C++ para capas de PyTorch";

    module.def(
        "relu_forward",
        &relu_forward_binding,
        py::arg("input"),
        "Ejecuta ReLU sobre un arreglo NumPy float32"
    );

    module.def(
        "tanh_forward",
        &tanh_forward_binding,
        py::arg("input"),
        "Ejecuta Tanh sobre un arreglo NumPy float32"
    );

    module.def(
        "linear_forward",
        &linear_forward_binding,
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        "Ejecuta una capa Linear usando arreglos NumPy float32"
    );

    module.def(
        "avg_pool2d_forward",
        &avg_pool2d_forward_binding,
        py::arg("input"),
        py::arg("output_height"),
        py::arg("output_width"),
        py::arg("kernel_height"),
        py::arg("kernel_width"),
        py::arg("stride_height"),
        py::arg("stride_width"),
        py::arg("padding_height"),
        py::arg("padding_width"),
        py::arg("count_include_pad") = true,
        py::arg("divisor_override") = py::none(),
        "Ejecuta pooling promedio 2D sobre un arreglo NumPy float32"
    );

    module.def(
        "conv2d_forward",
        &conv2d_forward_binding,
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("output_height"),
        py::arg("output_width"),
        py::arg("stride_height"),
        py::arg("stride_width"),
        py::arg("padding_height"),
        py::arg("padding_width"),
        "Ejecuta Conv2d para groups=1 y dilation=1"
    );
}