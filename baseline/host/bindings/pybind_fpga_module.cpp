// Permite trabajar con arrays NumPy desde Pybind11.
#include <pybind11/numpy.h>

// Permite crear el módulo Python mediante Pybind11.
#include <pybind11/pybind11.h>

// Incluye los wrappers C++ de cada capa FPGA.
#include "tanh_fpga.hpp"
#include "linear_fpga.hpp"
#include "avgpool2d_fpga.hpp"
#include "conv2d_fpga.hpp"
#include "depthwise_fpga.hpp"
#include "layeradd_fpga.hpp"
#include "pointwise_fpga.hpp"
#include "batchnorm2d_fpga.hpp"
#include "relu6_fpga.hpp"

#include "fpga_profile.hpp"

// Alias para utilizar el namespace de Pybind11.
namespace py = pybind11;

// Define un array NumPy de float32 y memoria contigua.
using FloatArray = py::array_t<
    float,
    py::array::c_style
>;


//Tanh
// Función que conecta Python con el wrapper C++ de Tanh.
py::array_t<float> tanh_fpga_forward_binding(
    const FloatArray& input
)
{
    // Obtiene la información del array NumPy.
    const py::buffer_info input_info =
        input.request();

    // Verifica que la entrada no esté vacía.
    if (input_info.size <= 0) {
        throw py::value_error(
            "La entrada no puede estar vacía"
        );
    }

    // Crea el array de salida con la misma forma que input.
    py::array_t<float> output(
        input_info.shape
    );

    // Obtiene la información del array de salida.
    py::buffer_info output_info =
        output.request();

    // Obtiene el puntero a los datos de entrada
    const auto* input_pointer =
        static_cast<const float*>(
            input_info.ptr
        );

    // Obtiene el puntero a los datos de salida
    auto* output_pointer =
        static_cast<float*>(
            output_info.ptr
        );

    // Llama al wrapper C++ de Tanh que ejecuta la operación en la FPGA.
    tanh_fpga_forward(
        input_pointer,
        output_pointer,
        static_cast<int>(input_info.size)
    );

    // Devuelve el resultado a Python.
    return output;
}

//Linear
// Función que conecta a Python con el wrapper C++ de Linear
py::array_t<float> linear_fpga_forward_binding(
    const FloatArray& input,
    const FloatArray& weight,
    const FloatArray& bias,
    bool use_bias
)
{   
    // Obtiene la información de los arrays NumPy.
    const py::buffer_info input_info =
        input.request();

    const py::buffer_info weight_info =
        weight.request();

    const py::buffer_info bias_info =
        bias.request();

    // Verifica la forma del input.
    if (input_info.ndim != 2) {
        throw py::value_error(
            "input debe tener forma [batch_size, in_features]"
        );
    }

    // Verfica la forma del weight.
    if (weight_info.ndim != 2) {
        throw py::value_error(
            "weight debe tener forma [out_features, in_features]"
        );
    }

    // Verifica la forma del bias.
    if (bias_info.ndim != 1) {
        throw py::value_error(
            "bias debe tener forma [out_features]"
        );
    }

    // Verfica que input y weight tengan los features compatibles
    if (input_info.shape[1] != weight_info.shape[1]) {
        throw py::value_error(
            "input.shape[1] debe coincidir con weight.shape[1]"
        );
    }

    // Verifica que bias coincida con los canales de salida.
    if (bias_info.shape[0] != weight_info.shape[0]) {
        throw py::value_error(
            "bias.shape[0] debe coincidir con weight.shape[0]"
        );
    }

    // Obtiene las dimensiones necesarias
    const int batch_size =
        static_cast<int>(input_info.shape[0]);

    const int in_features =
        static_cast<int>(input_info.shape[1]);

    const int out_features =
        static_cast<int>(weight_info.shape[0]);

   
    // Verifica que las dimensiones sean válidas.
    if (
        batch_size <= 0 ||
        in_features <= 0 ||
        out_features <= 0
    ) {
        throw py::value_error(
            "Las dimensiones deben ser mayores que cero"
        );
    }

    // Crea el array de salida.
    py::array_t<float> output(
        {
            batch_size,
            out_features
        }
    );

    // Obtiene la información del array de salida.
    py::buffer_info output_info =
        output.request();

    // Obtiene los punteros de los arrays.
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

    // Llama al wrapper de C++ de Linear
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

    // Devuelve el resultado a Python
    return output;
}

//AvgPool2d
//Función que conecta Python con el wrapper C++ de AvgPool2d.
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
    // Obtiene la información del arry Numpy.
    const py::buffer_info input_info =
        input.request();

    // Verfica que tenga 4 dimensiones
    if (input_info.ndim != 4) {
        throw py::value_error(
            "input debe tener forma "
            "[batch_size, channels, height, width]"
        );
    }

    // Obtiene las dimensiones del input.
    const int batch_size =
        static_cast<int>(input_info.shape[0]);

    const int input_channels =
        static_cast<int>(input_info.shape[1]);

    const int input_height =
        static_cast<int>(input_info.shape[2]);

    const int input_width =
        static_cast<int>(input_info.shape[3]);

    // Verfica que el tamaño del kernels sea válido.
    if (
        kernel_height <= 0 ||
        kernel_width <= 0
    ) {
        throw py::value_error(
            "kernel_size debe ser mayor que cero"
        );
    }

    // Verfica que el stride sea válido 
    if (
        stride_height <= 0 ||
        stride_width <= 0
    ) {
        throw py::value_error(
            "stride debe ser mayor que cero"
        );
    }

    // Verfica que el padding no sea negativo
    if (
        padding_height < 0 ||
        padding_width < 0
    ) {
        throw py::value_error(
            "padding no puede ser negativo"
        );
    }

    // Verfica el divisor personalizado
    if (
        use_divisor_override &&
        divisor_override <= 0
    ) {
        throw py::value_error(
            "divisor_override debe ser mayor que cero"
        );
    }

    // Calcula la altura de salida esperada
    const int output_height =
        (
            input_height
            + 2 * padding_height
            - kernel_height
        ) / stride_height
        + 1;

    // Calcula el ancho de salida esperado
    const int output_width =
        (
            input_width
            + 2 * padding_width
            - kernel_width
        ) / stride_width
        + 1;

    // Verifica que la configuración produzca una salida válida.
    if (
        output_height <= 0 ||
        output_width <= 0
    ) {
        throw py::value_error(
            "La configuración produce una salida vacía"
        );
    }

    // Crea el array Numpy de salida.
    py::array_t<float> output(
        {
            batch_size,
            input_channels,
            output_height,
            output_width
        }
    );

    // Obtiene la información de array de salida.
    py::buffer_info output_info =
        output.request();

    // Obtiene el puntero a los datos de entrada.
    const auto* input_pointer =
        static_cast<const float*>(
            input_info.ptr
        );

    // Obtiene el puntero a los datos de salida.
    auto* output_pointer =
        static_cast<float*>(
            output_info.ptr
        );

    // Llama al wrapper C++ que ejectua AvgPool2d.
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

    // Devuelve resultado a Python.
    return output;
}

//Conv2d
// Función que conecta Python con el wrapper C++ de Con
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
    // Obtiene la información de los arrays.
    const py::buffer_info input_info =
        input.request();

    const py::buffer_info weight_info =
        weight.request();

    const py::buffer_info bias_info =
        bias.request();

    // Verifica las dimensiones de los arrays.
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

    // Verfica compatibilidad entre input y weight. 
    if (input_info.shape[1] != weight_info.shape[1]) {
        throw py::value_error(
            "Los canales de input y weight no coinciden"
        );
    }

    // Verfica el tamaños del bias-
    if (bias_info.shape[0] != weight_info.shape[0]) {
        throw py::value_error(
            "El tamaño de bias no coincide con out_channels"
        );
    }

    // Verfica el stride.
    if (
        stride_height <= 0 ||
        stride_width <= 0
    ) {
        throw py::value_error(
            "stride debe ser mayor que cero"
        );
    }

    // Verfica el padding.
    if (
        padding_height < 0 ||
        padding_width < 0
    ) {
        throw py::value_error(
            "padding no puede ser negativo"
        );
    }

    // Obtiene las dimensiones de entrada.
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

    // Calcula las dimensiones de salida.
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

    // Verfica que la salida sea válida.
    if (
        output_height <= 0 ||
        output_width <= 0
    ) {
        throw py::value_error(
            "La configuración produce una salida vacía"
        );
    }

    // Crea el array de salida.
    py::array_t<float> output(
        {
            batch_size,
            output_channels,
            output_height,
            output_width
        }
    );

    // Obtiene la información del array de salida.
    py::buffer_info output_info =
        output.request();

    // Obtiene los punteros de los arrays.
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

    // Llama al wrapper de C++ de Conv2d.
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

    // Devuelve el resultado a Python.
    return output;
}

// ReLU6
// Función que conecta Python con el wrapper C++ RelU6
py::array_t<float> relu6_fpga_forward_binding(
    const FloatArray& input
)
{
    // Obtiene la información del array.
    const auto input_info = input.request();

    // Verfica que no esté vacío.
    if (input_info.size <= 0) {
        throw py::value_error(
            "La entrada no puede estar vacía"
        );
    }    

    // Crea la salida con las misma forma.
    py::array_t<float> output(input_info.shape);
    auto output_info = output.request();

    // Llama al wrapper C++
    relu6_fpga_forward(
        static_cast<const float*>(input_info.ptr),
        static_cast<float*>(output_info.ptr),
        static_cast<int>(input_info.size)
    );

    // Devuelve el resultado a Python.
    return output;
}

//LayerAdd
// Función que conecta Python con el wrapper de C++ LayerAdd.
py::array_t<float> layeradd_fpga_forward_binding(
    const FloatArray& input1,
    const FloatArray& input2,
    float alpha
)
{
    // Obtiene la información de las entradas.
    const auto input1_info = input1.request();
    const auto input2_info = input2.request();

    // Verfica que las entradas tenga la misma forma.
    if (input1_info.shape != input2_info.shape) {
        throw py::value_error(
            "input1 e input2 deben tener la misma forma"
        );
    }

    // Verfica que las entradas no estén vacías.
    if (input1_info.size <= 0) {
        throw py::value_error(
            "Las entradas no puden estar vacías"
        );
    }

    // Crea el array de salida.
    py::array_t<float> output(input1_info.shape);
    auto output_info = output.request();

    // Llama al wrapper C++
    layeradd_fpga_forward(
        static_cast<const float*>(input1_info.ptr),
        static_cast<const float*>(input2_info.ptr),
        static_cast<float*>(output_info.ptr),
        static_cast<int>(input1_info.size),
        alpha
    );

    // Devuelve el resultado.
    return output;
}

// Depthwise
// Función que conecta Python con el wrapper de C++ Depthwise.
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
    // Obtiene información de los arrays.
    const auto input_info = input.request();
    const auto weight_info = weight.request();
    const auto bias_info = bias.request();

    // Verficia las dimensiones del input.
    if (input_info.ndim != 4) {
        throw py::value_error(
            "input debe tener forma "
            "[batch_size, channels, height, width]"
        );
    }

    // Verifica las dimensiones del filtro.
    if (weight_info.ndim != 4) {
        throw py::value_error(
            "weight debe tener forma "
            "[channels, 1, kernel_height, kernel_width]"
        );
    }

    // Verfica la estructura del filtro.
    if (
        weight_info.shape[0] != input_info.shape[1] ||
        weight_info.shape[1] != 1
    ) {
        throw py::value_error(
            "weight debe contener un filtro por canal "
            "con forma [channels, 1, kernel_height, kernel_width]"
        );
    }

    // Verifica la dimensión del bias.
    if (bias_info.ndim != 1) {
        throw py::value_error(
            "bias debe tener forma [channels]"
        );
    }

    // Verifica que bias coincida con los canales.
    if (bias_info.shape[0] != input_info.shape[1]) {
        throw py::value_error(
            "El tamaño de bias debe coincidir con channels"
        );
    }

    // Verfica que input y weight no estén vacíos.
    if (
        input_info.size <= 0 ||
        weight_info.size <= 0
    ) {
        throw py::value_error(
            "input y weight no pueden estar vacíos"
        );
    }

    // Verifica el stride.
    if (
        stride_height <= 0 ||
        stride_width <= 0
    ) {
        throw py::value_error(
            "stride debe ser mayor que cero"
        );
    }

    // Verfica que el padding
    if (
        padding_height < 0 ||
        padding_width < 0
    ) {
        throw py::value_error(
            "padding no puede ser negativo"
        );
    }

    // Obtiene las dimensiones de entrada.
    const int batch_size =
        static_cast<int>(input_info.shape[0]);

    const int channels =
        static_cast<int>(input_info.shape[1]);

    const int input_height =
        static_cast<int>(input_info.shape[2]);

    const int input_width =
        static_cast<int>(input_info.shape[3]);

    // Obtiene el tamaño del kernel
    const int kernel_height =
        static_cast<int>(weight_info.shape[2]);

    const int kernel_width =
        static_cast<int>(weight_info.shape[3]);

    // Calcula las dimensiones con padding
    const int padded_height =
        input_height + 2 * padding_height;

    const int padded_width =
        input_width + 2 * padding_width;

    // Verifica que el filtro pueda aplicarse.
    if (
        kernel_height > padded_height ||
        kernel_width > padded_width
    ) {
        throw py::value_error(
            "El filtro no puede ser mayor que la entrada "
            "con padding"
        );
    }

    // Calcula las dimensiones de salida.
    const int output_height =
        (padded_height - kernel_height) / stride_height + 1;

    const int output_width =
        (padded_width - kernel_width) / stride_width + 1;

    // Crea el array de salida.
    py::array_t<float> output(
        {
            batch_size,
            channels,
            output_height,
            output_width
        }
    );

    auto output_info = output.request();

    // Llama al wrapper C++ de Depthwise.
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

    // Devuelve el resutado.
    return output;
}

// Pointwise
// Función que conecta Python con el wrapper C++ de Pointwise
py::array_t<float> pointwise_fpga_forward_binding(
    const FloatArray& input,
    const FloatArray& weight,
    const FloatArray& bias,
    bool use_bias
)
{
    // Obtiene la información de los arrays.
    const auto input_info = input.request();
    const auto weight_info = weight.request();
    const auto bias_info = bias.request();

    // Verifica las dimensiones del input.
    if (input_info.ndim != 4) {
        throw py::value_error(
            "input debe tener forma "
            "[batch_size, input_channels, height, width]"
        );
    }

    // Verifica las dimensiones del filtro.
    if (weight_info.ndim != 4) {
        throw py::value_error(
            "weight debe tener forma "
            "[output_channels, input_channels, 1, 1]"
        );
    }

    // Verifica que el filtro sea 1x1
    if (
        weight_info.shape[2] != 1 ||
        weight_info.shape[3] != 1
    ) {
        throw py::value_error(
            "Pointwise requiere filtros de 1 x 1"
        );
    }

    // Verifica los canales de entrada.
    if (input_info.shape[1] != weight_info.shape[1]) {
        throw py::value_error(
            "Los canales de input y weight no coinciden"
        );
    }

    // Verifica la dimensió del bias.
    if (bias_info.ndim != 1) {
        throw py::value_error(
            "bias debe tener forma [output_channels]"
        );
    }

    // Verifica el tamaño del bias.
    if (bias_info.shape[0] != weight_info.shape[0]) {
        throw py::value_error(
            "El tamaño de bias debe coincidir con output_channels"
        );
    }

    // Verifica que input y weigth no estén vacíos.
    if (
        input_info.size <= 0 ||
        weight_info.size <= 0
    ) {
        throw py::value_error(
            "input y weight no pueden estar vacíos"
        );
    }

    // Obtiene las dimensiones de entrada.
    const int batch_size =
        static_cast<int>(input_info.shape[0]);

    const int input_channels =
        static_cast<int>(input_info.shape[1]);

    const int input_height =
        static_cast<int>(input_info.shape[2]);

    const int input_width =
        static_cast<int>(input_info.shape[3]);

    // Obtiene la cantidad de canales de salida.
    const int output_channels =
        static_cast<int>(weight_info.shape[0]);

    // Crea el array de salida.
    py::array_t<float> output(
        {
            batch_size,
            output_channels,
            input_height,
            input_width
        }
    );

    auto output_info = output.request();

    // Llama al wrapper C++ de Pointwise.
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

    // Devuelve el resultado.
    return output;
}

// BatchNorm2d
// Función que conecta Python con el wrapper C++ de BatchNorm2D.
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
    // Obtiene la información de los arrays.
    const auto input_info = input.request();
    const auto weight_info = weight.request();
    const auto bias_info = bias.request();
    const auto running_mean_info = running_mean.request();
    const auto running_var_info = running_var.request();

    // Verifica las dimensiones del input.
    if (input_info.ndim != 4) {
        throw py::value_error(
            "input debe tener forma "
            "[batch_size, channels, height, width]"
        );
    }

    // Verifica que input no esté vacio
    if (input_info.size <= 0) {
        throw py::value_error(
            "input no puede estar vacío"
        );
    }

    //Verifica las dimensiones de los parámetros.
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

    // Verifica que todos los parámetros coincidad con channels.
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

    // Verifica el valor de epsilon.
    if (!(eps > 0.0f)) {
        throw py::value_error(
            "eps debe ser mayor que cero"
        );
    }

    // Obtiene las dimensiones de entrada.
    const int batch_size =
        static_cast<int>(input_info.shape[0]);

    const int channels =
        static_cast<int>(input_info.shape[1]);

    const int input_height =
        static_cast<int>(input_info.shape[2]);

    const int input_width =
        static_cast<int>(input_info.shape[3]);

    // Crea la salida con las misma forma que input.
    py::array_t<float> output(input_info.shape);

    auto output_info = output.request();

    // Llama al wrapper C++ de BatchNorm2D.
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

    // Devuelve el resultado.
    return output;
}

// Crea el módulo Python llamado "fpga_kernels"
PYBIND11_MODULE(fpga_kernels, module)
{
    //Para instrumentación
    module.def(
        "profile_begin",
        [](std::size_t capacity) {
            fpga_profile::begin(capacity);
        },
        py::arg("capacity") = 128
    );

    module.def(
        "profile_cancel",
        []() {
            fpga_profile::cancel();
        }
    );

    module.def(
        "profile_end",
        []() {
            auto& state = fpga_profile::state();

            // Desactiva el registro antes de construir objetos Python.
            state.enabled = false;

            py::dict result;
            py::list layers;

            for (const auto& record : state.records) {
                py::dict layer;

                layer["operation"] = record.operation;

                layer["preparacion_ms"] =
                    fpga_profile::milliseconds(
                        record.prepare_start,
                        record.to_device_start
                    );

                layer["sync_hacia_fpga_ms"] =
                    fpga_profile::milliseconds(
                        record.to_device_start,
                        record.to_device_end
                    );

                layer["lanzamiento_ms"] =
                    fpga_profile::milliseconds(
                        record.to_device_end,
                        record.launch_end
                    );

                layer["espera_ms"] =
                    fpga_profile::milliseconds(
                        record.launch_end,
                        record.wait_end
                    );

                layer["lanzamiento_y_espera_ms"] =
                    fpga_profile::milliseconds(
                        record.to_device_end,
                        record.wait_end
                    );

                layer["sync_hacia_cpu_ms"] =
                    fpga_profile::milliseconds(
                        record.wait_end,
                        record.from_device_end
                    );

                layer["copia_salida_ms"] =
                    fpga_profile::milliseconds(
                        record.from_device_end,
                        record.copy_end
                    );

                layer["latencia_capa_ms"] =
                    fpga_profile::milliseconds(
                        record.to_device_start,
                        record.from_device_end
                    );

                layer["total_region_host_ms"] =
                    fpga_profile::milliseconds(
                        record.prepare_start,
                        record.copy_end
                    );

                layers.append(layer);
            }

            result["layers"] = layers;
            result["operation_count"] = state.records.size();

            if (state.records.empty()) {
                result["latencia_red_ms"] = py::none();
            } else {
                result["latencia_red_ms"] =
                    fpga_profile::milliseconds(
                        state.records.front().to_device_start,
                        state.records.back().from_device_end
                    );
            }

            state.records.clear();

            return result;
        }
    );

    // Define una descripción del módulo.
    module.doc() =
        "Kernels HLS ejecutados en FPGA mediante XRT";

    //tanh
    // Expone tanh_forward para que se pueda llamar desde Python.
    module.def(
        "tanh_forward",
        &tanh_fpga_forward_binding,
        py::arg("input"),
        "Ejecuta Tanh en la FPGA sobre un arreglo NumPy float32"
    );

    // Linear
    // Expone linear_forward para Python.
    module.def(
        "linear_forward",
        &linear_fpga_forward_binding,
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("use_bias") = true,
        "Ejecuta Linear en la FPGA con arreglos NumPy float32"
    );	
   
    // AvgPool2d
    //Expone avgpool2d_forward para Python.
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

    //Conv2d
    // Expone conv2d_forward para Python.
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

    //ReLU6
    // Expone relu6_forward para Python.
    module.def(
        "relu6_forward",
        &relu6_fpga_forward_binding,
        py::arg("input"),
        "Ejecuta ReLU6 en la FPGA"
    );

    //LayerAdd
    // Expone laytera_forward para Python.
    module.def(
        "layeradd_forward",
        &layeradd_fpga_forward_binding,
        py::arg("input1"),
        py::arg("input2"),
        py::arg("alpha") = 1.0f,
        "Ejecuta LayerAdd en la FPGA con entradas de igual forma"
    );

    //Depthwise
    // Expone depthwise_forward para Python.
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

    //Pointwise
    // Expone pointwise_forward para Python.
    module.def(
        "pointwise_forward",
        &pointwise_fpga_forward_binding,
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("use_bias") = true,
        "Ejecuta Pointwise Conv2D en la FPGA con arreglos NumPy FP32"
    );

    //BatchNorm2d.
    // Expone batchnorm2d_forward para Python.
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
