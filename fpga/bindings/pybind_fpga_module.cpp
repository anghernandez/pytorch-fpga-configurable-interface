#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "tanh_fpga.hpp"
#include "linear_fpga.hpp"
#include "avgpool2d_fpga.hpp"
#include "conv2d_fpga.hpp"
#include "depthwise_fpga.hpp"
#include "layeradd_fpga.hpp"
#include "pointwise_fpga.hpp"
#include "batchnorm2d_fpga.hpp"
#include "relu6_fpga.hpp"

namespace py = pybind11;


using FloatArray = py::array_t<
    float,
    py::array::c_style
>;
/* esto crea un alias en lugar de escribir todo el tiempo
*  py::array_t<float, py::array::c_style>
*/

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

py::array_t<float> linear_fpga_forward_binding(
    const FloatArray& input,
    const FloatArray& weight,
    const FloatArray& bias,
    bool use_bias
)
/* esta función es el binding de Linear, es decir la capa que conecta el codigo
 * de C++ con la capa de Pytorch, su trabajo es recibir los datos de Python,
 * interpretarlos correctamente y entregarlos al wrapper de C++ linear_fpga_forward 
 */

{
    const py::buffer_info input_info =
        input.request();

    const py::buffer_info weight_info =
        weight.request();

    const py::buffer_info bias_info =
        bias.request();

    if (input_info.ndim != 2) {
        throw py::value_error(
            "input debe tener forma [batch_size, in_features]"
        );
    }

    if (weight_info.ndim != 2) {
        throw py::value_error(
            "weight debe tener forma [out_features, in_features]"
        );
    }

    if (bias_info.ndim != 1) {
        throw py::value_error(
            "bias debe tener forma [out_features]"
        );
    }

    if (input_info.shape[1] != weight_info.shape[1]) {
        throw py::value_error(
            "input.shape[1] debe coincidir con weight.shape[1]"
        );
    }

    if (bias_info.shape[0] != weight_info.shape[0]) {
        throw py::value_error(
            "bias.shape[0] debe coincidir con weight.shape[0]"
        );
    }

    const int batch_size =
        static_cast<int>(input_info.shape[0]);

    const int in_features =
        static_cast<int>(input_info.shape[1]);

    const int out_features =
        static_cast<int>(weight_info.shape[0]);

    /* se obtiene la info de lo arrays
    *  request() permite obtener información sobre el array NumPy.
    */
    
    if (
        batch_size <= 0 ||
        in_features <= 0 ||
        out_features <= 0
    ) {
        throw py::value_error(
            "Las dimensiones deben ser mayores que cero"
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

    const auto* bias_pointer =
        static_cast<const float*>(
            bias_info.ptr
        );

    auto* output_pointer =
        static_cast<float*>(
            output_info.ptr
        );

    linear_fpga_forward(
        input_pointer,
        weight_pointer,
        bias_pointer,
        output_pointer,
        batch_size,
        in_features,
        out_features,
        use_bias
    );

    return output;
}

py::array_t<float> avgpool2d_fpga_forward_binding(
    const FloatArray& input,
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
    const py::buffer_info input_info =
        input.request();

    if (input_info.ndim != 4) {
        throw py::value_error(
            "input debe tener forma "
            "[batch_size, channels, height, width]"
        );
    }

    const int batch_size =
        static_cast<int>(input_info.shape[0]);

    const int input_channels =
        static_cast<int>(input_info.shape[1]);

    const int input_height =
        static_cast<int>(input_info.shape[2]);

    const int input_width =
        static_cast<int>(input_info.shape[3]);

    if (
        kernel_height <= 0 ||
        kernel_width <= 0
    ) {
        throw py::value_error(
            "kernel_size debe ser mayor que cero"
        );
    }

    if (
        stride_height <= 0 ||
        stride_width <= 0
    ) {
        throw py::value_error(
            "stride debe ser mayor que cero"
        );
    }

    if (
        padding_height < 0 ||
        padding_width < 0
    ) {
        throw py::value_error(
            "padding no puede ser negativo"
        );
    }

    if (
        use_divisor_override &&
        divisor_override <= 0
    ) {
        throw py::value_error(
            "divisor_override debe ser mayor que cero"
        );
    }

    const int output_height =
        (
            input_height
            + 2 * padding_height
            - kernel_height
        ) / stride_height
        + 1;

    const int output_width =
        (
            input_width
            + 2 * padding_width
            - kernel_width
        ) / stride_width
        + 1;

    if (
        output_height <= 0 ||
        output_width <= 0
    ) {
        throw py::value_error(
            "La configuración produce una salida vacía"
        );
    }

    py::array_t<float> output(
        {
            batch_size,
            input_channels,
            output_height,
            output_width
        }
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

    avgpool2d_fpga_forward(
        input_pointer,
        output_pointer,
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

    return output;
}

py::array_t<float> conv2d_fpga_forward_binding(
    const FloatArray& input,
    const FloatArray& weight,
    const FloatArray& bias,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool use_bias
)
{
    const py::buffer_info input_info =
        input.request();

    const py::buffer_info weight_info =
        weight.request();

    const py::buffer_info bias_info =
        bias.request();

    if (input_info.ndim != 4) {
        throw py::value_error(
            "input debe tener forma "
            "[batch_size, in_channels, height, width]"
        );
    }

    if (weight_info.ndim != 4) {
        throw py::value_error(
            "weight debe tener forma "
            "[out_channels, in_channels, kernel_height, kernel_width]"
        );
    }

    if (bias_info.ndim != 1) {
        throw py::value_error(
            "bias debe tener forma [out_channels]"
        );
    }

    if (input_info.shape[1] != weight_info.shape[1]) {
        throw py::value_error(
            "Los canales de input y weight no coinciden"
        );
    }

    if (bias_info.shape[0] != weight_info.shape[0]) {
        throw py::value_error(
            "El tamaño de bias no coincide con out_channels"
        );
    }

    if (
        stride_height <= 0 ||
        stride_width <= 0
    ) {
        throw py::value_error(
            "stride debe ser mayor que cero"
        );
    }

    if (
        padding_height < 0 ||
        padding_width < 0
    ) {
        throw py::value_error(
            "padding no puede ser negativo"
        );
    }

    const int batch_size =
        static_cast<int>(input_info.shape[0]);

    const int input_channels =
        static_cast<int>(input_info.shape[1]);

    const int input_height =
        static_cast<int>(input_info.shape[2]);

    const int input_width =
        static_cast<int>(input_info.shape[3]);

    const int output_channels =
        static_cast<int>(weight_info.shape[0]);

    const int kernel_height =
        static_cast<int>(weight_info.shape[2]);

    const int kernel_width =
        static_cast<int>(weight_info.shape[3]);

    const int output_height =
        (
            input_height
            + 2 * padding_height
            - kernel_height
        ) / stride_height
        + 1;

    const int output_width =
        (
            input_width
            + 2 * padding_width
            - kernel_width
        ) / stride_width
        + 1;

    if (
        output_height <= 0 ||
        output_width <= 0
    ) {
        throw py::value_error(
            "La configuración produce una salida vacía"
        );
    }

    py::array_t<float> output(
        {
            batch_size,
            output_channels,
            output_height,
            output_width
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

    const auto* bias_pointer =
        static_cast<const float*>(
            bias_info.ptr
        );

    auto* output_pointer =
        static_cast<float*>(
            output_info.ptr
        );

    conv2d_fpga_forward(
        input_pointer,
        weight_pointer,
        bias_pointer,
        output_pointer,
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

    return output;
}

//relu6
py::array_t<float> relu6_fpga_forward_binding(
    const FloatArray& input
)
{
    const auto input_info = input.request();

    if (input_info.size <= 0) {
        throw py::value_error(
            "La entrada no puede estar vacía"
        );
    }    

    py::array_t<float> output(input_info.shape);
    auto output_info = output.request();

    relu6_fpga_forward(
        static_cast<const float*>(input_info.ptr),
        static_cast<float*>(output_info.ptr),
        static_cast<int>(input_info.size)
    );

    return output;
}


//layeradd
py::array_t<float> layeradd_fpga_forward_binding(
    const FloatArray& input1,
    const FloatArray& input2,
    float alpha
)
{
    const auto input1_info = input1.request();
    const auto input2_info = input2.request();

    if (input1_info.shape != input2_info.shape) {
        throw py::value_error(
            "input1 e input2 deben tener la misma forma"
        );
    }

    if (input1_info.size <= 0) {
        throw py::value_error(
            "Las entradas no puden estar vacías"
        );
    }

    py::array_t<float> output(input1_info.shape);
    auto output_info = output.request();

    layeradd_fpga_forward(
        static_cast<const float*>(input1_info.ptr),
        static_cast<const float*>(input2_info.ptr),
        static_cast<float*>(output_info.ptr),
        static_cast<int>(input1_info.size),
        alpha
    );

    return output;
}

// Depthwise
py::array_t<float> depthwise_fpga_forward_binding(
    const FloatArray& input,
    const FloatArray& weight,
    const FloatArray& bias,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool use_bias
)
{
    const auto input_info = input.request();
    const auto weight_info = weight.request();
    const auto bias_info = bias.request();

    if (input_info.ndim != 4) {
        throw py::value_error(
            "input debe tener forma "
            "[batch_size, channels, height, width]"
        );
    }

    if (weight_info.ndim != 4) {
        throw py::value_error(
            "weight debe tener forma "
            "[channels, 1, kernel_height, kernel_width]"
        );
    }

    if (
        weight_info.shape[0] != input_info.shape[1] ||
        weight_info.shape[1] != 1
    ) {
        throw py::value_error(
            "weight debe contener un filtro por canal "
            "con forma [channels, 1, kernel_height, kernel_width]"
        );
    }

    if (bias_info.ndim != 1) {
        throw py::value_error(
            "bias debe tener forma [channels]"
        );
    }

    if (bias_info.shape[0] != input_info.shape[1]) {
        throw py::value_error(
            "El tamaño de bias debe coincidir con channels"
        );
    }

    if (
        input_info.size <= 0 ||
        weight_info.size <= 0
    ) {
        throw py::value_error(
            "input y weight no pueden estar vacíos"
        );
    }

    if (
        stride_height <= 0 ||
        stride_width <= 0
    ) {
        throw py::value_error(
            "stride debe ser mayor que cero"
        );
    }

    if (
        padding_height < 0 ||
        padding_width < 0
    ) {
        throw py::value_error(
            "padding no puede ser negativo"
        );
    }

    const int batch_size =
        static_cast<int>(input_info.shape[0]);

    const int channels =
        static_cast<int>(input_info.shape[1]);

    const int input_height =
        static_cast<int>(input_info.shape[2]);

    const int input_width =
        static_cast<int>(input_info.shape[3]);

    const int kernel_height =
        static_cast<int>(weight_info.shape[2]);

    const int kernel_width =
        static_cast<int>(weight_info.shape[3]);

    const int padded_height =
        input_height + 2 * padding_height;

    const int padded_width =
        input_width + 2 * padding_width;

    if (
        kernel_height > padded_height ||
        kernel_width > padded_width
    ) {
        throw py::value_error(
            "El filtro no puede ser mayor que la entrada "
            "con padding"
        );
    }

    const int output_height =
        (padded_height - kernel_height) / stride_height + 1;

    const int output_width =
        (padded_width - kernel_width) / stride_width + 1;

    py::array_t<float> output(
        {
            batch_size,
            channels,
            output_height,
            output_width
        }
    );

    auto output_info = output.request();

    depthwise_fpga_forward(
        static_cast<const float*>(input_info.ptr),
        static_cast<const float*>(weight_info.ptr),
        static_cast<const float*>(bias_info.ptr),
        static_cast<float*>(output_info.ptr),
        batch_size,
        channels,
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
        use_bias
    );

    return output;
}

// Pointwise
py::array_t<float> pointwise_fpga_forward_binding(
    const FloatArray& input,
    const FloatArray& weight,
    const FloatArray& bias,
    bool use_bias
)
{
    const auto input_info = input.request();
    const auto weight_info = weight.request();
    const auto bias_info = bias.request();

    if (input_info.ndim != 4) {
        throw py::value_error(
            "input debe tener forma "
            "[batch_size, input_channels, height, width]"
        );
    }

    if (weight_info.ndim != 4) {
        throw py::value_error(
            "weight debe tener forma "
            "[output_channels, input_channels, 1, 1]"
        );
    }

    if (
        weight_info.shape[2] != 1 ||
        weight_info.shape[3] != 1
    ) {
        throw py::value_error(
            "Pointwise requiere filtros de 1 x 1"
        );
    }

    if (input_info.shape[1] != weight_info.shape[1]) {
        throw py::value_error(
            "Los canales de input y weight no coinciden"
        );
    }

    if (bias_info.ndim != 1) {
        throw py::value_error(
            "bias debe tener forma [output_channels]"
        );
    }

    if (bias_info.shape[0] != weight_info.shape[0]) {
        throw py::value_error(
            "El tamaño de bias debe coincidir con output_channels"
        );
    }

    if (
        input_info.size <= 0 ||
        weight_info.size <= 0
    ) {
        throw py::value_error(
            "input y weight no pueden estar vacíos"
        );
    }

    const int batch_size =
        static_cast<int>(input_info.shape[0]);

    const int input_channels =
        static_cast<int>(input_info.shape[1]);

    const int input_height =
        static_cast<int>(input_info.shape[2]);

    const int input_width =
        static_cast<int>(input_info.shape[3]);

    const int output_channels =
        static_cast<int>(weight_info.shape[0]);

    py::array_t<float> output(
        {
            batch_size,
            output_channels,
            input_height,
            input_width
        }
    );

    auto output_info = output.request();

    pointwise_fpga_forward(
        static_cast<const float*>(input_info.ptr),
        static_cast<const float*>(weight_info.ptr),
        static_cast<const float*>(bias_info.ptr),
        static_cast<float*>(output_info.ptr),
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_channels,
        use_bias
    );

    return output;
}


// BatchNorm2d
py::array_t<float> batchnorm2d_fpga_forward_binding(
    const FloatArray& input,
    const FloatArray& weight,
    const FloatArray& bias,
    const FloatArray& running_mean,
    const FloatArray& running_var,
    float eps,
    bool affine,
    bool use_bias
)
{
    const auto input_info = input.request();
    const auto weight_info = weight.request();
    const auto bias_info = bias.request();
    const auto running_mean_info = running_mean.request();
    const auto running_var_info = running_var.request();

    if (input_info.ndim != 4) {
        throw py::value_error(
            "input debe tener forma "
            "[batch_size, channels, height, width]"
        );
    }

    if (input_info.size <= 0) {
        throw py::value_error(
            "input no puede estar vacío"
        );
    }

    if (
        weight_info.ndim != 1 ||
        bias_info.ndim != 1 ||
        running_mean_info.ndim != 1 ||
        running_var_info.ndim != 1
    ) {
        throw py::value_error(
            "weight, bias, running_mean y running_var "
            "deben tener forma [channels]"
        );
    }

    if (
        weight_info.shape[0] != input_info.shape[1] ||
        bias_info.shape[0] != input_info.shape[1] ||
        running_mean_info.shape[0] != input_info.shape[1] ||
        running_var_info.shape[0] != input_info.shape[1]
    ) {
        throw py::value_error(
            "El tamaño de weight, bias, running_mean y "
            "running_var debe coincidir con channels"
        );
    }

    if (!(eps > 0.0f)) {
        throw py::value_error(
            "eps debe ser mayor que cero"
        );
    }

    const int batch_size =
        static_cast<int>(input_info.shape[0]);

    const int channels =
        static_cast<int>(input_info.shape[1]);

    const int input_height =
        static_cast<int>(input_info.shape[2]);

    const int input_width =
        static_cast<int>(input_info.shape[3]);

    py::array_t<float> output(input_info.shape);

    auto output_info = output.request();

    batchnorm2d_fpga_forward(
        static_cast<const float*>(input_info.ptr),
        static_cast<const float*>(weight_info.ptr),
        static_cast<const float*>(bias_info.ptr),
        static_cast<const float*>(running_mean_info.ptr),
        static_cast<const float*>(running_var_info.ptr),
        static_cast<float*>(output_info.ptr),
        batch_size,
        channels,
        input_height,
        input_width,
        eps,
        affine,
        use_bias
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

    module.def(
        "linear_forward",
        &linear_fpga_forward_binding,
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("use_bias") = true,
        "Ejecuta Linear en la FPGA con arreglos NumPy float32"
    );	
   
    module.def(
        "avgpool2d_forward",
        &avgpool2d_fpga_forward_binding,
        py::arg("input"),
        py::arg("kernel_height"),
        py::arg("kernel_width"),
        py::arg("stride_height"),
        py::arg("stride_width"),
        py::arg("padding_height") = 0,
        py::arg("padding_width") = 0,
        py::arg("count_include_pad") = true,
        py::arg("use_divisor_override") = false,
        py::arg("divisor_override") = 0,
        "Ejecuta AvgPool2D en la FPGA sobre un arreglo NumPy FP32"
    );

    module.def(
        "conv2d_forward",
        &conv2d_fpga_forward_binding,
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("stride_height") = 1,
        py::arg("stride_width") = 1,
        py::arg("padding_height") = 0,
        py::arg("padding_width") = 0,
        py::arg("use_bias") = true,
        "Ejecuta Conv2D en la FPGA con arreglos NumPy FP32"
    );


    module.def(
        "relu6_forward",
        &relu6_fpga_forward_binding,
        py::arg("input"),
        "Ejecuta ReLU6 en la FPGA"
    );

    module.def(
        "layeradd_forward",
        &layeradd_fpga_forward_binding,
        py::arg("input1"),
        py::arg("input2"),
        py::arg("alpha") = 1.0f,
        "Ejecuta LayerAdd en la FPGA con entradas de igual forma"
    );

        module.def(
        "depthwise_forward",
        &depthwise_fpga_forward_binding,
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("stride_height") = 1,
        py::arg("stride_width") = 1,
        py::arg("padding_height") = 0,
        py::arg("padding_width") = 0,
        py::arg("use_bias") = true,
        "Ejecuta Depthwise Conv2D en la FPGA con arreglos NumPy FP32"
    );

        module.def(
        "pointwise_forward",
        &pointwise_fpga_forward_binding,
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("use_bias") = true,
        "Ejecuta Pointwise Conv2D en la FPGA con arreglos NumPy FP32"
    );

        module.def(
        "batchnorm2d_forward",
        &batchnorm2d_fpga_forward_binding,
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("running_mean"),
        py::arg("running_var"),
        py::arg("eps") = 1e-5f,
        py::arg("affine") = true,
        py::arg("use_bias") = true,
        "Ejecuta BatchNorm2D en la FPGA con estadísticas almacenadas"
    );


}
