// Pybind11: interfaz mínima para el grafo diferido FpgaRuntime.

#include "fpga_runtime.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <xrt/xrt/xrt_kernel.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

// ============================================================
// Utilidad interna: grupo de memoria correspondiente a un kernel.
// El llamador aporta nombre y posición del argumento XRT.
// ============================================================
static int memory_group(const std::string& kernel_name, int argument_index)
{
    if (argument_index < 0)
        throw std::invalid_argument("El índice del argumento no puede ser negativo");
    auto& runtime = FpgaRuntime::instance();
    xrt::kernel kernel(runtime.device(), runtime.uuid(), kernel_name);
    return kernel.group_id(argument_index);
}

// ============================================================
// Módulo independiente durante la migración a grafo diferido.
// ============================================================
PYBIND11_MODULE(fpga_dag, module)
{
    module.doc() = "Interfaz Python del DAG diferido para FPGA/XRT";

    // Construcción del grafo: no ejecuta kernels.
    module.def("clear_graph", []() {
        FpgaRuntime::instance().clear_graph();
    });
    module.def("add_input", [](const std::string& id) {
        FpgaRuntime::instance().add_graph_input(id);
    });
    module.def("add_parameter", [](const std::string& id) {
        FpgaRuntime::instance().add_graph_parameter(id);
    });
    module.def("add_operation", [](
        const std::string& operation,
        const std::vector<std::string>& inputs,
        const std::vector<std::string>& parameters,
        const std::string& output,
        const std::unordered_map<std::string, int>& int_args,
        const std::unordered_map<std::string, float>& float_args) {
        OperationNode node;
        node.operation = operation;
        node.inputs = inputs;
        node.parameters = parameters;
        node.output = output;
        node.int_args = int_args;
        node.float_args = float_args;
        FpgaRuntime::instance().add_operation(node);
    }, py::arg("operation"), py::arg("inputs"), py::arg("parameters"),
       py::arg("output"), py::arg("int_args") = std::unordered_map<std::string, int>{},
       py::arg("float_args") = std::unordered_map<std::string, float>{});
    module.def("set_outputs", [](const std::vector<std::string>& ids) {
        FpgaRuntime::instance().set_graph_outputs(ids);
    });
    module.def("finalize_graph", []() {
        FpgaRuntime::instance().finalize_graph();
    });
    module.def("consumers", [](const std::string& id) {
        return FpgaRuntime::instance().graph_consumers(id);
    });

    // Carga explícita CPU -> buffer XRT, solo en los límites del grafo.
    // El buffer de entrada se consumirá al ejecutar el DAG.
    module.def("upload_input", [](
        const std::string& id, const FloatArray& data,
        const std::string& kernel_name, int argument_index) {
        const auto info = data.request();
        if (info.size <= 0 || static_cast<std::size_t>(info.size) >
                              std::numeric_limits<std::size_t>::max() / sizeof(float))
            throw std::invalid_argument("Entrada vacía o tamaño inválido");
        auto& runtime = FpgaRuntime::instance();
        auto& bo = runtime.create_buffer(id,
            static_cast<std::size_t>(info.size) * sizeof(float),
            memory_group(kernel_name, argument_index));
        std::copy_n(static_cast<const float*>(info.ptr),
                    static_cast<std::size_t>(info.size), bo.map<float*>());
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }, py::arg("id"), py::arg("data"), py::arg("kernel_name"),
       py::arg("argument_index") = 0);

    // Parámetros persistentes: se transfieren una vez y permanecen en XRT.
    module.def("load_parameter", [](
        const std::string& id, const FloatArray& data,
        const std::string& kernel_name, int argument_index) {
        const auto info = data.request();
        if (info.size <= 0)
            throw std::invalid_argument("El parámetro no puede estar vacío");
        FpgaRuntime::instance().load_parameter(
            id, static_cast<const float*>(info.ptr),
            static_cast<std::size_t>(info.size),
            memory_group(kernel_name, argument_index));
    }, py::arg("id"), py::arg("data"), py::arg("kernel_name"),
       py::arg("argument_index"));

    // Ejecuta todos los nodos registrados; los wrappers esperan run.wait().
    module.def("execute_graph", []() {
        FpgaRuntime::instance().execute_graph();
    });

    // Copia única FPGA -> CPU al recuperar una salida final.
    // Shape es explícito para que NumPy reconstruya la forma correcta.
    module.def("download_output", [](
        const std::string& id, const std::vector<py::ssize_t>& shape,
        bool release) -> py::array_t<float> {
        if (shape.empty())
            throw std::invalid_argument("La salida debe tener dimensiones");
        std::size_t elements = 1;
        for (const auto dim : shape) {
            if (dim <= 0 || elements > std::numeric_limits<std::size_t>::max() /
                                         static_cast<std::size_t>(dim))
                throw std::invalid_argument("Forma inválida para la salida");
            elements *= static_cast<std::size_t>(dim);
        }
        auto& runtime = FpgaRuntime::instance();
        auto& bo = runtime.buffer(id);
        if (elements > bo.size() / sizeof(float))
            throw std::invalid_argument("La forma excede el tamaño del buffer");
        py::array_t<float> result(shape);
        bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::copy_n(bo.map<float*>(), elements,
                    static_cast<float*>(result.request().ptr));
        if (release)
            runtime.consume_buffer(id); // Reserva de salida calculada por el DAG.
        return result;
    }, py::arg("id"), py::arg("shape"), py::arg("release") = true);

    module.def("release_output", [](const std::string& id) {
        FpgaRuntime::instance().consume_buffer(id);
    });
    module.def("has_buffer", [](const std::string& id) {
        return FpgaRuntime::instance().has_buffer(id);
    });
    module.def("buffer_count", []() {
        return FpgaRuntime::instance().buffer_count();
    });
    module.def("reuse_count", []() {
        return FpgaRuntime::instance().reuse_count();
    });
}