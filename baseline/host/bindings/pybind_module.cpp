/*#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fpga_runtime.hpp"

#include "../Pointwise/SW/pointwise_fpga.hpp"
#include "../ReLU6/SW/relu6_fpga.hpp"

namespace py = pybind11;

using FloatArray = py::array_t<float, py::array::c_style>;


// ============================================================
// Tensor residente en FPGA
// ============================================================
//
// Este objeto representa memoria que permanece en FPGA.
//
// Python no conoce:
//   - xrt::bo
//   - memory_group
//   - sync()
//   - map()
//   - memcpy()
//
// Solo conoce un FpgaTensor.
//

class FpgaTensor
{
public:

    FpgaBuffer buffer;

    std::vector<py::ssize_t> shape;

    std::size_t bytes;


    FpgaTensor(
        FpgaBuffer&& buffer_,
        std::vector<py::ssize_t> shape_
    )
        : buffer(std::move(buffer_)),
          shape(std::move(shape_)),
          bytes(buffer.size)
    {
    }


    ~FpgaTensor()
    {
        try
        {
            FpgaRuntime::instance().release_buffer(buffer);
        }
        catch (...)
        {
            // No lanzar excepciones desde un destructor.
        }
    }


    FpgaTensor(const FpgaTensor&) = delete;
    FpgaTensor& operator=(const FpgaTensor&) = delete;


    std::vector<py::ssize_t> get_shape() const
    {
        return shape;
    }


    std::size_t get_bytes() const
    {
        return bytes;
    }


    // --------------------------------------------------------
    // FPGA -> CPU
    //
    // Esta función representa la salida del modelo.
    // --------------------------------------------------------

    py::array_t<float> to_numpy() const
    {
        py::array_t<float> output(shape);

        auto output_info = output.request();

        FpgaRuntime::instance().download_buffer(
            const_cast<FpgaBuffer&>(buffer),
            output_info.ptr,
            bytes
        );

        return output;
    }
};


// ============================================================
// Crear tensor FPGA desde NumPy
// ============================================================
//
// Esta operación es una entrada CPU -> FPGA.
//
// Para la prueba inicial usamos el grupo de entrada de
// Pointwise.
//
// Más adelante esta función se puede generalizar sin cambiar
// FpgaTensor.
//

std::shared_ptr<FpgaTensor> fpga_from_numpy(
    const FloatArray& input
)
{
    const auto info = input.request();

    if (info.size <= 0)
    {
        throw std::invalid_argument(
            "El tensor de entrada no puede estar vacío"
        );
    }


    const std::size_t bytes =
        static_cast<std::size_t>(info.size) *
        sizeof(float);


    auto& runtime = FpgaRuntime::instance();


    // --------------------------------------------------------
    // El grupo se obtiene desde el kernel.
    // Python nunca conoce este valor.
    // --------------------------------------------------------

    xrt::kernel kernel(
        runtime.device(),
        runtime.uuid(),
        "pointwise_conv2d_forward"
    );


    const std::uint32_t memory_group =
        kernel.group_id(0);


    FpgaBuffer buffer =
        runtime.acquire_buffer(
            bytes,
            memory_group
        );


    runtime.upload_buffer(
        buffer,
        info.ptr,
        bytes
    );


    std::vector<py::ssize_t> shape(
        info.shape.begin(),
        info.shape.end()
    );


    return std::make_shared<FpgaTensor>(
        std::move(buffer),
        std::move(shape)
    );
}


// ============================================================
// Crear buffer de pesos de Pointwise
// ============================================================

std::shared_ptr<FpgaTensor> pointwise_weight_from_numpy(
    const FloatArray& weight
)
{
    const auto info = weight.request();

    if (info.size <= 0)
    {
        throw std::invalid_argument(
            "Los pesos no pueden estar vacíos"
        );
    }


    if (info.ndim != 4)
    {
        throw std::invalid_argument(
            "Los pesos de Pointwise deben ser 4D"
        );
    }


    if (info.shape[2] != 1 ||
        info.shape[3] != 1)
    {
        throw std::invalid_argument(
            "Pointwise requiere pesos con kernel 1x1"
        );
    }


    const std::size_t bytes =
        static_cast<std::size_t>(info.size) *
        sizeof(float);


    auto& runtime = FpgaRuntime::instance();


    xrt::kernel kernel(
        runtime.device(),
        runtime.uuid(),
        "pointwise_conv2d_forward"
    );


    const std::uint32_t memory_group =
        kernel.group_id(1);


    FpgaBuffer buffer =
        runtime.acquire_buffer(
            bytes,
            memory_group
        );


    runtime.upload_buffer(
        buffer,
        info.ptr,
        bytes
    );


    std::vector<py::ssize_t> shape(
        info.shape.begin(),
        info.shape.end()
    );


    return std::make_shared<FpgaTensor>(
        std::move(buffer),
        std::move(shape)
    );
}


// ============================================================
// Crear buffer de bias de Pointwise
// ============================================================

std::shared_ptr<FpgaTensor> pointwise_bias_from_numpy(
    const FloatArray& bias
)
{
    const auto info = bias.request();

    if (info.size <= 0)
    {
        throw std::invalid_argument(
            "El bias no puede estar vacío"
        );
    }


    if (info.ndim != 1)
    {
        throw std::invalid_argument(
            "El bias de Pointwise debe ser 1D"
        );
    }


    const std::size_t bytes =
        static_cast<std::size_t>(info.size) *
        sizeof(float);


    auto& runtime = FpgaRuntime::instance();


    xrt::kernel kernel(
        runtime.device(),
        runtime.uuid(),
        "pointwise_conv2d_forward"
    );


    const std::uint32_t memory_group =
        kernel.group_id(2);


    FpgaBuffer buffer =
        runtime.acquire_buffer(
            bytes,
            memory_group
        );


    runtime.upload_buffer(
        buffer,
        info.ptr,
        bytes
    );


    std::vector<py::ssize_t> shape(
        info.shape.begin(),
        info.shape.end()
    );


    return std::make_shared<FpgaTensor>(
        std::move(buffer),
        std::move(shape)
    );
}


// ============================================================
// Pointwise sobre buffers FPGA
// ============================================================
//
// IMPORTANTE:
//
// input, weight y bias ya están en FPGA.
//
// No:
//   - upload
//   - download
//   - map
//   - memcpy
//
// La salida también queda en FPGA.
//

std::shared_ptr<FpgaTensor> pointwise_forward_buffer(
    const std::shared_ptr<FpgaTensor>& input,
    const std::shared_ptr<FpgaTensor>& weight,
    const std::shared_ptr<FpgaTensor>& bias,
    bool use_bias
)
{
    if (!input)
    {
        throw std::invalid_argument(
            "input es nulo"
        );
    }


    if (!weight)
    {
        throw std::invalid_argument(
            "weight es nulo"
        );
    }


    if (!bias)
    {
        throw std::invalid_argument(
            "bias es nulo"
        );
    }


    if (input->shape.size() != 4)
    {
        throw std::invalid_argument(
            "Pointwise espera input 4D"
        );
    }


    if (weight->shape.size() != 4)
    {
        throw std::invalid_argument(
            "Pointwise espera weight 4D"
        );
    }


    if (weight->shape[2] != 1 ||
        weight->shape[3] != 1)
    {
        throw std::invalid_argument(
            "Pointwise requiere kernel 1x1"
        );
    }


    if (bias->shape.size() != 1)
    {
        throw std::invalid_argument(
            "Pointwise espera bias 1D"
        );
    }


    const int batch_size =
        static_cast<int>(input->shape[0]);

    const int input_channels =
        static_cast<int>(input->shape[1]);

    const int input_height =
        static_cast<int>(input->shape[2]);

    const int input_width =
        static_cast<int>(input->shape[3]);

    const int output_channels =
        static_cast<int>(weight->shape[0]);


    if (input_channels !=
        static_cast<int>(weight->shape[1]))
    {
        throw std::invalid_argument(
            "Los canales de input y weight no coinciden"
        );
    }


    if (output_channels !=
        static_cast<int>(bias->shape[0]))
    {
        throw std::invalid_argument(
            "Los canales de output y bias no coinciden"
        );
    }


    const std::size_t output_elements =
        static_cast<std::size_t>(batch_size) *
        static_cast<std::size_t>(output_channels) *
        static_cast<std::size_t>(input_height) *
        static_cast<std::size_t>(input_width);


    const std::size_t output_bytes =
        output_elements * sizeof(float);


    auto& runtime = FpgaRuntime::instance();


    xrt::kernel kernel(
        runtime.device(),
        runtime.uuid(),
        "pointwise_conv2d_forward"
    );


    const std::uint32_t output_group =
        kernel.group_id(3);


    FpgaBuffer output_buffer =
        runtime.acquire_buffer(
            output_bytes,
            output_group
        );


    pointwise_fpga_forward_buffer(
        input->buffer,
        weight->buffer,
        bias->buffer,
        output_buffer,
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_channels,
        use_bias
    );


    std::vector<py::ssize_t> output_shape = {
        batch_size,
        output_channels,
        input_height,
        input_width
    };


    return std::make_shared<FpgaTensor>(
        std::move(output_buffer),
        std::move(output_shape)
    );
}


// ============================================================
// ReLU6 sobre buffer FPGA
// ============================================================
//
// input ya está en FPGA.
// output queda en FPGA.
//

std::shared_ptr<FpgaTensor> relu6_forward_buffer(
    const std::shared_ptr<FpgaTensor>& input
)
{
    if (!input)
    {
        throw std::invalid_argument(
            "input es nulo"
        );
    }


    auto& runtime = FpgaRuntime::instance();


    xrt::kernel kernel(
        runtime.device(),
        runtime.uuid(),
        "relu6_forward"
    );


    const std::uint32_t output_group =
        kernel.group_id(1);


    FpgaBuffer output_buffer =
        runtime.acquire_buffer(
            input->bytes,
            output_group
        );


    const int size =
        static_cast<int>(
            input->bytes / sizeof(float)
        );


    relu6_fpga_forward_buffer(
        input->buffer,
        output_buffer,
        size
    );


    return std::make_shared<FpgaTensor>(
        std::move(output_buffer),
        input->shape
    );
}


// ============================================================
// Módulo
// ============================================================

PYBIND11_MODULE(fpga_buffer, module)
{
    module.doc() =
        "Bindings PyBind11 para ejecucion FPGA mediante buffers";


    // --------------------------------------------------------
    // FpgaTensor
    // --------------------------------------------------------

    py::class_<
        FpgaTensor,
        std::shared_ptr<FpgaTensor>
    >(module, "FpgaTensor")

        .def(
            "to_numpy",
            &FpgaTensor::to_numpy,
            "Descarga el tensor FPGA a NumPy"
        )

        .def_property_readonly(
            "shape",
            &FpgaTensor::get_shape
        )

        .def_property_readonly(
            "bytes",
            &FpgaTensor::get_bytes
        );


    // --------------------------------------------------------
    // Entrada CPU -> FPGA
    // --------------------------------------------------------

    module.def(
        "from_numpy",
        &fpga_from_numpy,
        py::arg("input"),
        "Copia un tensor NumPy a FPGA"
    );


    // --------------------------------------------------------
    // Parámetros Pointwise
    // --------------------------------------------------------

    module.def(
        "pointwise_weight_from_numpy",
        &pointwise_weight_from_numpy,
        py::arg("weight"),
        "Carga los pesos de Pointwise en FPGA"
    );


    module.def(
        "pointwise_bias_from_numpy",
        &pointwise_bias_from_numpy,
        py::arg("bias"),
        "Carga el bias de Pointwise en FPGA"
    );


    // --------------------------------------------------------
    // Pointwise
    // --------------------------------------------------------

    module.def(
        "pointwise_forward",
        &pointwise_forward_buffer,
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("use_bias") = true,
        "Ejecuta Pointwise sobre buffers FPGA"
    );


    // --------------------------------------------------------
    // ReLU6
    // --------------------------------------------------------

    module.def(
        "relu6_forward",
        &relu6_forward_buffer,
        py::arg("input"),
        "Ejecuta ReLU6 sobre un buffer FPGA"
    );
}*/