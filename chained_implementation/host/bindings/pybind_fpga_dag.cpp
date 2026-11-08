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

// Simplifica el acceso a pybind11 y define arreglos NumPy
// float32, contiguos en memoria, para intercambiar datos
namespace py = pybind11;
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

// ------------------------------------------------------------
// Obtiene el grupo de memoria asociado a un argumento
// específico de un kernel XRT
// ------------------------------------------------------------
static int memory_group(const std::string& kernel_name, int argument_index)
{
    if (argument_index < 0)
        throw std::invalid_argument("El índice del argumento no puede ser negativo");

    auto& runtime = FpgaRuntime::instance();

    auto& kernel = runtime.get_kernel(kernel_name);

    // Consulta la conectividad de memoria del argumento del kernel.
    return kernel.group_id(argument_index);
}

// ------------------------------------------------------------
// Expone las funciones del runtime C++ como un módulo Python
// denominado fpga_dag
// ------------------------------------------------------------
PYBIND11_MODULE(fpga_dag, module)
{
    module.doc() = "Interfaz Python del DAG diferido para FPGA/XRT";

    
    // ------------------------------------------------------------
    // Expone las funciones para registrar entradas, parámetros,
    // operaciones y salidas del DAG sin ejecutar kernels
    // ------------------------------------------------------------
    module.def("clear_graph", []() {
        FpgaRuntime::instance().clear_graph();
    });
    module.def("add_input", [](const std::string& id) {
        FpgaRuntime::instance().add_graph_input(id);
    });
    module.def("add_parameter", [](const std::string& id) {
        FpgaRuntime::instance().add_graph_parameter(id);
    });

    // Construye un OperationNode con los argumentos recibidos
    // desde Python y lo registra en FpgaRuntime
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

    // Valida y finaliza el DAG para permitir su ejecución
    module.def("finalize_graph", []() {
        FpgaRuntime::instance().finalize_graph();
    });
    module.def("consumers", [](const std::string& id) {
        return FpgaRuntime::instance().graph_consumers(id);
    });

    // ------------------------------------------------------------
    // Recibe un arreglo NumPy, asigna un BO y transfiere
    // sus datos desde el host hacia la FPGA.
    // ------------------------------------------------------------
    module.def("upload_input", [](
        const std::string& id, const FloatArray& data,
        const std::string& kernel_name, int argument_index) {
        
        // Obtiene la cantidad de elementos y el puntero a los datos
        const auto info = data.request();
        if (info.size <= 0 || static_cast<std::size_t>(info.size) >
                              std::numeric_limits<std::size_t>::max() / sizeof(float))
            throw std::invalid_argument("Entrada vacía o tamaño inválido");
        auto& runtime = FpgaRuntime::instance();

        // Asigna el BO utilizando el grupo de memoria del kernel.
        auto& bo = runtime.create_buffer(id,
            static_cast<std::size_t>(info.size) * sizeof(float),
            memory_group(kernel_name, argument_index));

        // Copia los datos de NumPy a la memoria mapeada del BO
        std::copy_n(static_cast<const float*>(info.ptr),
                    static_cast<std::size_t>(info.size), bo.map<float*>());
        
        // Sincroniza los datos de entrada hacia el dispositivo.
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }, py::arg("id"), py::arg("data"), py::arg("kernel_name"),
       py::arg("argument_index") = 0);

    // ------------------------------------------------------------
    // Transfiere los parámetros del modelo a buffers persistentes
    // para reutilizarlos entre ejecuciones
    // ------------------------------------------------------------
    module.def("load_parameter", [](
        const std::string& id, const FloatArray& data,
        const std::string& kernel_name, int argument_index) {
        const auto info = data.request();
        if (info.size <= 0)
            throw std::invalid_argument("El parámetro no puede estar vacío");

        // Delega la carga y persistencia del parámetro al runtime
        FpgaRuntime::instance().load_parameter(
            id, static_cast<const float*>(info.ptr),
            static_cast<std::size_t>(info.size),
            memory_group(kernel_name, argument_index));
    }, py::arg("id"), py::arg("data"), py::arg("kernel_name"),
       py::arg("argument_index"));


    // ------------------------------------------------------------
    // Ejecuta el DAG mediante FpgaRuntime y libera el GIL
    // para permitir que otros hilos de Python continúen
    // ------------------------------------------------------------
    module.def("execute_graph", []() {
        auto& runtime = FpgaRuntime::instance();

        // Libera temporalmente el GIL durante la ejecución en C++
        py::gil_scoped_release release;
        runtime.execute_graph();
    });



    // ------------------------------------------------------------
    // Ejecutar varios batches sobre el mismo grafo
    // ------------------------------------------------------------
    // Cada mapa relaciona:
    // nombre simbólico de entrada -> identificador del buffer cargado.
    //
    // Devuelve un mapa por batch con las salidas conservadas:
    // nombre simbólico de salida -> identificador de su buffer.
    module.def("execute_pipeline", [](
        const std::vector<
            std::unordered_map<std::string, std::string>
        >& batch_inputs
    ) {
        if (batch_inputs.empty()) {
            throw std::invalid_argument(
                "El pipeline necesita al menos un batch"
            );
        }

        auto& runtime = FpgaRuntime::instance();

        // 4. Crea un estado independiente para cada batch (ejecición pipeline)
        std::vector<PipelineBatchState> batches(batch_inputs.size());

        for (std::size_t b = 0; b < batch_inputs.size(); ++b) {

            // Asocia las entradas simbólicas con los BO de cada batch
            batches[b].tensor_buffers = batch_inputs[b];
        }

        {
            // Liberar el bloqueo de Python durante la ejecución,
            // permitiendo que avance el hilo de monitoreo.
            py::gil_scoped_release release;

            // Ejecuta el pipeline sin mantener bloqueado el GIL de Python
            // 5. Entrega al runtime las entradas y los estados de los batches (ejecución pipeline)
            runtime.execute_pipeline(batches);
        }

        // Prepara los mapas de buffers disponibles al finalizar
        std::vector<
            std::unordered_map<std::string, std::string>
        > outputs(batches.size());

        for (std::size_t b = 0; b < batches.size(); ++b) {
            for (const auto& entry : batches[b].tensor_buffers) {

                // Conserva los identificadores de buffers que siguen activos.
                if (runtime.has_buffer(entry.second)) {
                    outputs[b].emplace(entry.first, entry.second);
                }
            }
        }

        // Devuelve a Python los identificadores de buffers por batch.
        return outputs;
    }, py::arg("batch_inputs"));

    
    // ------------------------------------------------------------
    // Recupera un resultado desde la FPGA y lo convierte
    // en un arreglo NumPy con la forma indicada.
    // ------------------------------------------------------------
    module.def("download_output", [](
        const std::string& id, const std::vector<py::ssize_t>& shape,
        bool release) -> py::array_t<float> {
        if (shape.empty())
            throw std::invalid_argument("La salida debe tener dimensiones");
        std::size_t elements = 1;

        // Valida la forma solicitada y calcula la cantidad de elementos.
        for (const auto dim : shape) {
            if (dim <= 0 || elements > std::numeric_limits<std::size_t>::max() /
                                         static_cast<std::size_t>(dim))
                throw std::invalid_argument("Forma inválida para la salida");
            elements *= static_cast<std::size_t>(dim);
        }
        auto& runtime = FpgaRuntime::instance();
        
        // Recupera el BO correspondiente al resultado solicitado
        auto& bo = runtime.buffer(id);
        if (elements > bo.size() / sizeof(float))
            throw std::invalid_argument("La forma excede el tamaño del buffer");
        py::array_t<float> result(shape);

        // Sincroniza el resultado desde el dispositivo hacia el host
        bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        // Copia los valores del BO al arreglo NumPy
        std::copy_n(bo.map<float*>(), elements,
                    static_cast<float*>(result.request().ptr));

        // Libera el consumo reservado para la salida, si se solicita
        if (release)
            runtime.consume_buffer(id); // Reserva de salida calculada por el DAG.
        return result;
    }, py::arg("id"), py::arg("shape"), py::arg("release") = true);


    // ------------------------------------------------------------
    // Expone funciones para liberar buffers y consultar
    // su disponibilidad y estadísticas de reutilización.
    // ------------------------------------------------------------
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

    // Solicita la liberación explícita de un BO sin consumidores
    module.def("release_buffer", [](const std::string& id) {
        FpgaRuntime::instance().release_buffer(id);
    }, py::arg("id"));
    
    // Expone el cierre y limpieza de recursos del runtime
    module.def("shutdown", []() {
    FpgaRuntime::instance().shutdown();
});
}