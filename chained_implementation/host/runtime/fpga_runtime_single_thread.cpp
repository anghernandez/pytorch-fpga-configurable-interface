// Variante reconstruida del planificador anterior compartido en la conversación.
// No es una copia byte a byte: conserva el algoritmo de un hilo y la interfaz actual.
// Runtime: administra la memoria y la ejecución del grafo en la FPGA.
// XRT (Xilinx Runtime) permite controlar la FPGA desde la CPU.
// BO (Buffer Object) es el objeto XRT que representa un buffer de datos.
#include "fpga_runtime.hpp"

#include "tanh_fpga.hpp"
#include "relu6_fpga.hpp"
#include "linear_fpga.hpp"
#include "conv2d_fpga.hpp"
#include "avgpool2d_fpga.hpp"
#include "pointwise_fpga.hpp"
#include "depthwise_fpga.hpp"
#include "batchnorm2d_fpga.hpp"
#include "layeradd_fpga.hpp"


#include <algorithm>
#include <iostream>
#include <xrt/xrt/xrt_kernel.h>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <string>
#include <utility>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <initializer_list>

// ------------------------------------------------------------
// 1. Instancia compartida e inicialización
// ------------------------------------------------------------
// Singleton: devuelve la misma instancia del runtime en cada llamada.
FpgaRuntime& FpgaRuntime::instance()
{
    // Se construye en la primera llamada y se conserva para las siguientes.
    static FpgaRuntime instance;
    return instance;
}

// Abre el dispositivo 0 y carga el xclbin, el binario de los kernels FPGA.
FpgaRuntime::FpgaRuntime() : device_(0)
{
    const std::string xclbin_path =
        "/lib/firmware/xilinx/fpga_accelerators/fpga_kernels.xclbin";
    std::cout << "[FpgaRuntime] Cargando: " << xclbin_path << std::endl;
    // Guarda el identificador del binario cargado para abrir sus kernels.
    uuid_ = device_.load_xclbin(xclbin_path);
}

// Permiten consultar el dispositivo y el UUID (identificador del xclbin).
xrt::device& FpgaRuntime::device() { return device_; }
const xrt::uuid& FpgaRuntime::uuid() const { return uuid_; }

// ------------------------------------------------------------
// 2. Reserva, persistencia y reutilización de buffers
// ------------------------------------------------------------
// Asocia un ID a un BO reutilizable o reserva uno nuevo.
xrt::bo& FpgaRuntime::create_buffer(const std::string& id,
                                    std::size_t bytes,
                                    int memory_group)
{
    if (id.empty())
        throw std::invalid_argument("El identificador no puede estar vacío");
    if (bytes == 0)
        throw std::invalid_argument("El tamaño del buffer debe ser positivo");
    if (has_buffer(id))
        throw std::invalid_argument("Ya existe un buffer: " + id);

    // El pool guarda BO libres, este runtime reutiliza solo los del mismo
    // grupo de memoria y con capacidad suficiente para los bytes solicitados.
    for (auto it = pool_.begin(); it != pool_.end(); ++it) {
        if (it->first == memory_group && it->second.size() >= bytes) {
            // Pasa el BO al registro de buffers activos sin copiar los datos del tensor.
            buffers_.emplace(id, std::move(it->second));
            pool_.erase(it);
            memory_groups_.emplace(id, memory_group);
            ++reuse_count_;
            return buffers_.at(id);
        }
    }

    // Si el pool no ofrece un BO compatible, lo construye dentro de buffers_.
    auto result = buffers_.try_emplace(id, device_, bytes, memory_group);
    memory_groups_.emplace(id, memory_group);
    return result.first->second;
}

// Carga parámetros float32 y los conserva para reutilizarlos entre ejecuciones
xrt::bo& FpgaRuntime::load_parameter(const std::string& id,
                                      const float* values,
                                      std::size_t count,
                                      int memory_group)
{
    if (graph_running_)
        throw std::logic_error("No se pueden cargar parámetros durante la ejecución");
    if (values == nullptr || count == 0 ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::invalid_argument("Datos o tamaño de parámetro inválidos");

    xrt::bo& bo = create_buffer(id, count * sizeof(float), memory_group);
    try {
        // map() permite escribir desde la CPU en la parte del BO accesible al host.
        std::copy_n(values, count, bo.map<float*>());

        // Sincroniza lo escrito para que la FPGA pueda leer los parámetros.
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        // Persistente: no se recicla automáticamente al terminar una operación.
        configure_lifetime(id, 0, true);
    } catch (...) {
        // Revierte la carga incompleta y vuelve a propagar el error.
        release_buffer(id);
        throw;
    }
    return bo;
}

// Devuelve el BO activo asociado al ID; falla si ese ID no existe.
xrt::bo& FpgaRuntime::buffer(const std::string& id)
{
    auto it = buffers_.find(id);
    if (it == buffers_.end())
        throw std::out_of_range("Buffer no encontrado: " + id);
    return it->second;
}

// Comprueba si el ID pertenece al registro de buffers activos.
bool FpgaRuntime::has_buffer(const std::string& id) const
{
    return buffers_.find(id) != buffers_.end();
}

// Registra cuántos usos quedan y si el BO debe conservarse de forma persistente.
void FpgaRuntime::configure_lifetime(const std::string& id,
                                     std::size_t consumers,
                                     bool persistent)
{
    if (!has_buffer(id))
        throw std::out_of_range("Buffer no encontrado: " + id);
    if (!persistent && consumers == 0)
        throw std::invalid_argument("Un temporal debe tener consumidores");
    if (remaining_consumers_.count(id))
        throw std::logic_error("El ciclo de vida ya fue configurado: " + id);

    remaining_consumers_[id] = consumers;
    persistent_[id] = persistent;
}

// Descuenta un uso terminado y recicla el BO temporal cuando ya no quedan usos.
void FpgaRuntime::consume_buffer(const std::string& id)
{
    if (!has_buffer(id))
        throw std::out_of_range("Buffer no encontrado: " + id);
    auto it = remaining_consumers_.find(id);
    if (it == remaining_consumers_.end())
        throw std::logic_error("No se registraron consumidores: " + id);
    if (it->second == 0)
        throw std::logic_error("Se consumió más veces de lo registrado: " + id);

    --it->second;
    if (it->second == 0 && !persistent_.at(id))
        release_buffer(id);
}

// Retira el ID activo y guarda el BO en el pool para reutilizar su memoria.
// Debe llamarse después de que terminen los kernels que usan ese BO.
void FpgaRuntime::release_buffer(const std::string& id)
{
    auto it = buffers_.find(id);
    if (it == buffers_.end())
        throw std::out_of_range("Buffer no encontrado: " + id);

    auto pending = remaining_consumers_.find(id);
    if (pending != remaining_consumers_.end() && pending->second != 0)
        throw std::logic_error("Buffer con consumidores pendientes: " + id);

    const int group = memory_groups_.at(id);
    // Conserva el BO junto con su grupo; borra únicamente su registro anterior.
    pool_.emplace_back(group, std::move(it->second));
    buffers_.erase(it);
    memory_groups_.erase(id);
    remaining_consumers_.erase(id);
    persistent_.erase(id);
}

// Consultas: cantidad de BO activos, BO libres y reutilizaciones acumuladas.
std::size_t FpgaRuntime::buffer_count() const { return buffers_.size(); }
std::size_t FpgaRuntime::pooled_buffer_count() const { return pool_.size(); }
std::size_t FpgaRuntime::reuse_count() const { return reuse_count_; }
// Descarta los BO libres conservados para reutilización.
void FpgaRuntime::clear_pool() { pool_.clear(); }

// ------------------------------------------------------------
// 3. Registro y validación del grafo
// ------------------------------------------------------------
// Borra la descripción del grafo; esta función no elimina los BO existentes.
void FpgaRuntime::clear_graph()
{
    if (graph_running_)
        throw std::logic_error("No se puede borrar un grafo en ejecución");
    graph_nodes_.clear();
    graph_inputs_.clear();
    graph_parameters_.clear();
    graph_outputs_.clear();
    graph_order_.clear();
    graph_consumers_.clear();
    graph_finalized_ = false;
}

// Declara un tensor de entrada por su ID; todavía no carga sus datos.
void FpgaRuntime::add_graph_input(const std::string& id)
{
    if (graph_finalized_)
        throw std::logic_error("El grafo ya fue finalizado");
    if (id.empty() || graph_parameters_.count(id) || !graph_inputs_.insert(id).second)
        throw std::invalid_argument("Entrada de grafo inválida o repetida: " + id);
}

// Declara el ID de un parámetro que deberá estar cargado antes de ejecutar.
void FpgaRuntime::add_graph_parameter(const std::string& id)
{
    if (graph_finalized_)
        throw std::logic_error("El grafo ya fue finalizado");
    if (id.empty() || graph_inputs_.count(id) || !graph_parameters_.insert(id).second)
        throw std::invalid_argument("Parámetro de grafo inválido o repetido: " + id);
}

// Guarda un OperationNode: operación, IDs de tensores y argumentos del kernel.
// El nodo describe el trabajo; registrarlo todavía no lo ejecuta.
void FpgaRuntime::add_operation(const OperationNode& node)
{
    if (graph_finalized_)
        throw std::logic_error("El grafo ya fue finalizado");
    if (node.operation.empty() || node.output.empty() || node.inputs.empty())
        throw std::invalid_argument("La operación necesita nombre, entradas y salida");
    graph_nodes_.push_back(node);
}

// Indica qué tensores deben conservarse como resultados finales.
void FpgaRuntime::set_graph_outputs(const std::vector<std::string>& ids)
{
    if (graph_finalized_)
        throw std::logic_error("El grafo ya fue finalizado");
    if (ids.empty())
        throw std::invalid_argument("El grafo necesita al menos una salida");
    graph_outputs_ = ids;
}

// Valida el DAG (grafo dirigido sin ciclos), ordena sus nodos y cuenta los usos
// de cada tensor. El orden respeta las dependencias entre operaciones.
void FpgaRuntime::finalize_graph()
{
    if (graph_finalized_)
        throw std::logic_error("El grafo ya fue finalizado");
    if (graph_outputs_.empty())
        throw std::logic_error("Primero configure las salidas del grafo");

    // Mapa tensor → nodo productor. Se prepara localmente para no guardar
    // un resultado parcial si alguna validación falla.

    std::unordered_map<std::string, std::size_t> producer;
    for (std::size_t i = 0; i < graph_nodes_.size(); ++i) {
        const auto& id = graph_nodes_[i].output;
        if (graph_inputs_.count(id) || graph_parameters_.count(id) ||
            !producer.emplace(id, i).second)
            throw std::invalid_argument("Tensor producido más de una vez: " + id);
    }

    // dependents[i]: nodos que necesitan la salida del nodo i.
    // indegree[i]: dependencias que faltan para poder ordenar ese nodo.
    // consumers[id]: cantidad de usos del tensor registrada en el grafo.
    std::vector<std::vector<std::size_t>> dependents(graph_nodes_.size());
    std::vector<std::size_t> indegree(graph_nodes_.size(), 0);
    std::unordered_map<std::string, std::size_t> consumers;

    for (std::size_t i = 0; i < graph_nodes_.size(); ++i) {
        const auto& node = graph_nodes_[i];
        for (const auto& input : node.inputs) {
            if (!graph_inputs_.count(input) && !producer.count(input))
                throw std::invalid_argument("Entrada sin productor: " + input);
            // Cada aparición de una entrada cuenta, aunque se repita en un mismo nodo.
            ++consumers[input];
            if (const auto it = producer.find(input); it != producer.end()) {
                dependents[it->second].push_back(i);
                ++indegree[i];
            }
        }
        for (const auto& param : node.parameters) {
            if (!graph_parameters_.count(param))
                throw std::invalid_argument("Parámetro no registrado: " + param);
        }
    }

    std::unordered_set<std::string> unique_outputs;
    for (const auto& id : graph_outputs_) {
        if (!graph_inputs_.count(id) && !producer.count(id))
            throw std::invalid_argument("Salida sin productor: " + id);
        if (!unique_outputs.insert(id).second)
            throw std::invalid_argument("Salida del grafo repetida: " + id);
        // Añade una reserva para que el resultado final sobreviva hasta su descarga.
        ++consumers[id];
    }

    // Cola de índices de nodos sin dependencias pendientes: el primero en entrar
    // es el primero en salir. size_t se utiliza aquí como tipo de índice.
    std::queue<std::size_t> ready;
    for (std::size_t i = 0; i < indegree.size(); ++i)
        if (indegree[i] == 0) ready.push(i);

    std::vector<std::size_t> order;
    // Ordenamiento topológico: añade un nodo y desbloquea sus sucesores.
    // Aquí solo se calcula el orden; todavía no se ejecutan kernels.
    while (!ready.empty()) {
        const auto i = ready.front();
        ready.pop();
        order.push_back(i);
        for (const auto successor : dependents[i])
            if (--indegree[successor] == 0) ready.push(successor);
    }
    // Si quedan nodos sin ordenar, existe una dependencia circular.
    if (order.size() != graph_nodes_.size())
        throw std::invalid_argument("El grafo contiene un ciclo");

    // Guarda el orden y los contadores solo después de validar todo el grafo.
    graph_order_ = std::move(order);
    graph_consumers_ = std::move(consumers);
    graph_finalized_ = true;
}

// Permite consultar los nodos registrados sin copiar el vector.
const std::vector<OperationNode>& FpgaRuntime::graph_nodes() const
{
    return graph_nodes_;
}

// Devuelve los índices de los nodos en el orden válido de ejecución.
const std::vector<std::size_t>& FpgaRuntime::graph_execution_order() const
{
    if (!graph_finalized_)
        throw std::logic_error("El grafo no fue finalizado");
    return graph_order_;
}

// Consulta los usos calculados al finalizar el grafo, incluida la reserva
// de una salida final; no es el contador de usos restantes durante la ejecución.
std::size_t FpgaRuntime::graph_consumers(const std::string& id) const
{
    if (!graph_finalized_)
        throw std::logic_error("El grafo no fue finalizado");
    const auto it = graph_consumers_.find(id);
    return it == graph_consumers_.end() ? 0 : it->second;
}

// ------------------------------------------------------------
// 4. Ciclo de vida de los buffers en la ejecución secuencial
// ------------------------------------------------------------
// Verifica los BO iniciales y prepara sus contadores antes de ejecutar.
void FpgaRuntime::begin_graph_execution()
{
    if (!graph_finalized_ || graph_running_)
        throw std::logic_error("El DAG debe estar finalizado e inactivo");

    // Exige entradas cargadas y sin contadores de una ejecución anterior.
    for (const auto& id : graph_inputs_) {
        if (!has_buffer(id))
            throw std::logic_error("Falta el buffer de entrada: " + id);
        if (remaining_consumers_.count(id))
            throw std::logic_error("Entrada con ciclo de vida previo: " + id);
    }
    for (const auto& id : graph_parameters_) {
        if (!has_buffer(id))
            throw std::logic_error("Falta el buffer de parámetro: " + id);
        if (remaining_consumers_.count(id) && !persistent_.at(id))
            throw std::logic_error("Parámetro no persistente: " + id);
    }
    for (const auto& node : graph_nodes_)
        if (has_buffer(node.output))
            throw std::logic_error("Salida ya asignada antes de ejecutar: " + node.output);

    // Los parámetros se conservan; las entradas se gestionan según sus usos.
    for (const auto& id : graph_parameters_)
        if (!remaining_consumers_.count(id))
            configure_lifetime(id, 0, true);
    for (const auto& id : graph_inputs_)
        configure_lifetime(id, graph_consumers(id), graph_consumers(id) == 0);

    // Posición del siguiente nodo dentro de graph_order_, empezando por el primero.
    graph_next_node_ = 0;
    graph_running_ = true;
}

// Reserva la salida del siguiente nodo y registra cuántos usos tendrá.
xrt::bo& FpgaRuntime::create_graph_output_buffer(const std::string& id,
                                                  std::size_t bytes,
                                                  int memory_group)
{
    if (!graph_running_ || graph_next_node_ >= graph_order_.size())
        throw std::logic_error("No hay un nodo pendiente de ejecución");
    const auto& expected = graph_nodes_.at(graph_order_[graph_next_node_]);
    if (id != expected.output)
        throw std::logic_error("La salida no corresponde al siguiente nodo: " + id);

    auto& bo = create_buffer(id, bytes, memory_group);
    configure_lifetime(id, graph_consumers(id));
    return bo;
}

// Después de terminar el kernel, consume sus entradas y avanza al siguiente nodo.
void FpgaRuntime::complete_graph_operation(std::size_t node_index)
{
    if (!graph_running_ || graph_next_node_ >= graph_order_.size() ||
        graph_order_[graph_next_node_] != node_index)
        throw std::logic_error("El nodo no sigue el orden del DAG");

    const auto& node = graph_nodes_.at(node_index);
    if (!has_buffer(node.output))
        throw std::logic_error("El nodo no tiene buffer de salida: " + node.output);
    for (const auto& id : node.inputs)
        if (!has_buffer(id))
            throw std::logic_error("Falta buffer de entrada: " + id);

    // Descuenta cada argumento de entrada; una entrada repetida cuenta varias veces.
    for (const auto& id : node.inputs)
        consume_buffer(id);
    ++graph_next_node_;
}

// Cierra la ejecución si todos los nodos terminaron. Las salidas conservan
// su reserva hasta descargarlas y llamar a consume_buffer(id).
void FpgaRuntime::end_graph_execution()
{
    if (!graph_running_ || graph_next_node_ != graph_order_.size())
        throw std::logic_error("La ejecución del DAG está incompleta");
    graph_running_ = false;
    graph_next_node_ = 0;
}

// ------------------------------------------------------------
// 5. Ejecución secuencial del grafo
// ------------------------------------------------------------
// Ejecuta un nodo a la vez: cada wrapper espera a que termine su kernel.
void FpgaRuntime::execute_graph()
{
    if (!graph_finalized_ || graph_running_)
        throw std::logic_error("El DAG debe estar finalizado e inactivo");

    // Comprueba las operaciones admitidas y la cantidad de entradas y parámetros.
    for (const auto& node : graph_nodes_) {
        const auto& name = node.operation;
        if (name != "ReLU6" && name != "Tanh" && name != "LayerAdd" &&
            name != "Linear" && name != "Conv2D" && name != "AvgPool2D" &&
            name != "Pointwise" && name != "Depthwise" && name != "BatchNorm2D")
            throw std::invalid_argument("Operación FPGA no soportada: " + name);
        const std::size_t input_count = name == "LayerAdd" ? 2 : 1;
        const std::size_t parameter_count =
            name == "BatchNorm2D" ? 4 :
            (name == "Linear" || name == "Conv2D" ||
             name == "Pointwise" || name == "Depthwise") ? 2 : 0;
        if (node.inputs.size() != input_count ||
            node.parameters.size() != parameter_count)
            throw std::invalid_argument("Número de entradas o parámetros incorrecto: " + name);
    }

    try {
    begin_graph_execution();
    for (const auto index : graph_order_) {
        const auto& node = graph_nodes_.at(index);
        const auto& name = node.operation;
        // Funciones locales (lambdas) para leer argumentos del nodo actual.
        // [&] permite acceder por referencia a las variables del contexto.
        const auto arg = [&](const char* key) -> int {
            const auto it = node.int_args.find(key);
            if (it == node.int_args.end())
                throw std::invalid_argument("Falta argumento " + std::string(key) + " en " + name);
            return it->second;
        };
        const auto farg = [&](const char* key) -> float {
            const auto it = node.float_args.find(key);
            if (it == node.float_args.end())
                throw std::invalid_argument("Falta argumento " + std::string(key) + " en " + name);
            return it->second;
        };
        // Resuelve los IDs de entradas y parámetros como referencias a sus BO.
        const auto input = [&](std::size_t i) -> xrt::bo& {
            return buffer(node.inputs.at(i));
        };
        const auto parameter = [&](std::size_t i) -> xrt::bo& {
            return buffer(node.parameters.at(i));
        };
        // Calcula los bytes de la salida float32 y reserva su BO.
        // Rechaza dimensiones no positivas y multiplicaciones que desborden size_t.
        const auto output = [&](std::initializer_list<int> dimensions,
                                const char* kernel_name,
                                int output_index) -> xrt::bo& {
            std::size_t count = 1;
            for (const int dim : dimensions) {
                if (dim <= 0 || count > std::numeric_limits<std::size_t>::max() /
                                             static_cast<std::size_t>(dim))
                    throw std::invalid_argument("Dimensión inválida en " + name);
                count *= static_cast<std::size_t>(dim);
            }
            if (count > std::numeric_limits<std::size_t>::max() / sizeof(float))
                throw std::overflow_error("Buffer demasiado grande");

            auto& kernel = get_kernel(kernel_name);

            // Obtiene el grupo de memoria conectado al argumento de salida del kernel.
            const int group = kernel.group_id(output_index);

            auto& bo = create_graph_output_buffer(
                node.output,
                count * sizeof(float),
                group
            );

            return bo;
        };

        // Selecciona el wrapper de cada operación y le entrega sus BO y argumentos.
        // El último número de output() es el índice del argumento de salida del kernel.
        if (name == "ReLU6") {
            const int size = arg("size");
            auto& out = output({size}, "relu6_forward", 1);
            relu6_fpga_forward(input(0), out, size);
        } else if (name == "Tanh") {
            const int size = arg("size");
            auto& out = output({size}, "tanh_forward", 1);
            tanh_fpga_forward(input(0), out, size);
        } else if (name == "LayerAdd") {
            const int size = arg("size");
            const float alpha = farg("alpha");
            auto& out = output({size}, "layer_add_forward", 2);
            layeradd_fpga_forward(input(0), input(1), out, size, alpha);
        } else if (name == "Linear") {
            const int batch = arg("batch_size"), in = arg("in_features");
            const int out_features = arg("out_features");
            auto& out = output({batch, out_features}, "linear_forward", 3);
            linear_fpga_forward(input(0), parameter(0), parameter(1), out,
                                batch, in, out_features, arg("use_bias") != 0);
        } else if (name == "Conv2D") {
            const int b = arg("batch_size"), ic = arg("input_channels");
            const int ih = arg("input_height"), iw = arg("input_width");
            const int oc = arg("output_channels"), oh = arg("output_height");
            const int ow = arg("output_width");
            auto& out = output({b, oc, oh, ow}, "conv2d_forward", 3);
            conv2d_fpga_forward(input(0), parameter(0), parameter(1), out,
                b, ic, ih, iw, oc, oh, ow, arg("kernel_height"),
                arg("kernel_width"), arg("stride_height"), arg("stride_width"),
                arg("padding_height"), arg("padding_width"), arg("use_bias") != 0);
        } else if (name == "AvgPool2D") {
            const int b = arg("batch_size"), c = arg("input_channels");
            const int ih = arg("input_height"), iw = arg("input_width");
            const int oh = arg("output_height"), ow = arg("output_width");
            auto& out = output({b, c, oh, ow}, "avg_pool2d_forward", 1);
            avgpool2d_fpga_forward(input(0), out, b, c, ih, iw, oh, ow,
                arg("kernel_height"), arg("kernel_width"),
                arg("stride_height"), arg("stride_width"),
                arg("padding_height"), arg("padding_width"),
                arg("count_include_pad") != 0, arg("use_divisor_override") != 0,
                arg("divisor_override"));
        } else if (name == "Pointwise") {
            const int b = arg("batch_size"), ic = arg("input_channels");
            const int ih = arg("input_height"), iw = arg("input_width");
            const int oc = arg("output_channels");
            auto& out = output({b, oc, ih, iw}, "pointwise_conv2d_forward", 3);
            pointwise_fpga_forward(input(0), parameter(0), parameter(1), out,
                b, ic, ih, iw, oc, arg("use_bias") != 0);
        } else if (name == "Depthwise") {
            const int b = arg("batch_size"), c = arg("channels");
            const int ih = arg("input_height"), iw = arg("input_width");
            const int oh = arg("output_height"), ow = arg("output_width");
            auto& out = output({b, c, oh, ow}, "depthwise_conv2d_forward", 3);
            depthwise_fpga_forward(input(0), parameter(0), parameter(1), out,
                b, c, ih, iw, oh, ow, arg("kernel_height"), arg("kernel_width"),
                arg("stride_height"), arg("stride_width"),
                arg("padding_height"), arg("padding_width"), arg("use_bias") != 0);
        } else if (name == "BatchNorm2D") {
            const int b = arg("batch_size"), c = arg("channels");
            const int h = arg("input_height"), w = arg("input_width");
            const float eps = farg("eps");
            auto& out = output({b, c, h, w}, "batchnorm2d_forward", 5);
            batchnorm2d_fpga_forward(input(0), parameter(0), parameter(1),
                parameter(2), parameter(3), out,
                b, c, h, w, eps, arg("affine") != 0, arg("use_bias") != 0);
        }
        // El wrapper ya terminó: ahora pueden descontarse los usos de sus entradas.
        complete_graph_operation(index);

        // Registra un instante; en este archivo no se utiliza para calcular una duración.
        const auto node_end = std::chrono::steady_clock::now();

    }
    end_graph_execution();
    } catch (...) {
        // Ante un error, descarta las entradas y salidas parciales de esta ejecución.
        // Conserva los parámetros y la descripción del grafo.
        for (const auto& id : graph_inputs_) {
            buffers_.erase(id);
            memory_groups_.erase(id);
            remaining_consumers_.erase(id);
            persistent_.erase(id);
        }
        for (const auto& node : graph_nodes_) {
            buffers_.erase(node.output);
            memory_groups_.erase(node.output);
            remaining_consumers_.erase(node.output);
            persistent_.erase(node.output);
        }
        // Descarta también los BO ya reciclados y restablece el estado de ejecución.
        pool_.clear();
        graph_running_ = false;
        graph_next_node_ = 0;
        throw;
    }
}

// ------------------------------------------------------------
// 6. Pipeline de varios batches
// ------------------------------------------------------------
// Cada batch ejecuta el mismo grafo con activaciones propias y parámetros
// compartidos, permite solapar kernels distintos, con un trabajo activo por nombre
// admite ReLU6, Tanh, Linear, Conv2D, AvgPool2D, Pointwise, Depthwise,
// BatchNorm2D y LayerAdd.
void FpgaRuntime::execute_pipeline(
    std::vector<PipelineBatchState>& batches
)
{
    if (!graph_finalized_ || graph_running_) {
        throw std::logic_error(
            "El grafo debe estar finalizado e inactivo"
        );
    }

    if (batches.empty() || graph_nodes_.empty()) {
        throw std::invalid_argument(
            "El pipeline necesita batches y operaciones"
        );
    }

    // Plan de un nodo compartido por todos los batches; describe su kernel y salida.

    struct NodePlan {
        std::string kernel_name;  // Nombre del kernel en el xclbin.
        std::size_t output_bytes;  // Capacidad necesaria para su salida.
        int output_argument;  // Índice del argumento de salida, contando desde 0.
    };

    // plans[i] corresponde al mismo índice i de graph_nodes_.
    std::vector<NodePlan> plans;
    plans.reserve(graph_nodes_.size());

    // Mapa tensor → nodo productor, utilizado para comprobar las dependencias
    std::unordered_map<std::string, std::size_t> producers;

    // Obtiene un argumento entero obligatorio; rechaza claves ausentes
    const auto required_int = [](const OperationNode& node,
                                 const char* key) -> int {
        const auto it = node.int_args.find(key);

        if (it == node.int_args.end()) {
            throw std::invalid_argument(
                "Falta argumento " + std::string(key) +
                " en " + node.operation
            );
        }

        return it->second;
    };

    // Obtiene un argumento flotante y rechaza NaN (valor no numérico) o infinito
    const auto required_float = [](const OperationNode& node,
                                   const char* key) -> float {
        const auto it = node.float_args.find(key);
        if (it == node.float_args.end() || !std::isfinite(it->second)) {
            throw std::invalid_argument(
                "Argumento flotante ausente o no finito: " +
                std::string(key) + " en " + node.operation
            );
        }
        return it->second;
    };

    // Multiplica las dimensiones y sizeof(float) para calcular los bytes
    // Comprueba cada multiplicación para evitar desbordamientos.
    const auto checked_tensor_bytes =
        [](std::initializer_list<int> dimensions) {
            std::size_t count = 1;

            const auto maximum =
                std::numeric_limits<std::size_t>::max();

            for (const int dimension : dimensions) {
                if (dimension <= 0) {
                    throw std::invalid_argument(
                        "Las dimensiones deben ser mayores que cero"
                    );
                }

                const auto value =
                    static_cast<std::size_t>(dimension);

                if (count > maximum / value) {
                    throw std::overflow_error(
                        "Cantidad de elementos demasiado grande"
                    );
                }

                count *= value;
            }

            if (count > maximum / sizeof(float)) {
                throw std::overflow_error(
                    "Tamaño de buffer demasiado grande"
                );
            }

            return count * sizeof(float);
        };

    // Todos los parámetros deben estar cargados y marcados como persistentes.
    // Los batches comparten estos BO
    for (const auto& id : graph_parameters_) {
        const auto lifetime = remaining_consumers_.find(id);
        const auto persistence = persistent_.find(id);

        if (!has_buffer(id) ||
            lifetime == remaining_consumers_.end() ||
            persistence == persistent_.end() ||
            !persistence->second) {
            throw std::logic_error(
                "Falta un parametro persistente: " + id
            );
        }
    }

    // Valida cada operación y prepara su kernel, tamaño de salida e índice de argumento.
    for (std::size_t i = 0; i < graph_nodes_.size(); ++i) {
        const auto& node = graph_nodes_[i];

        // LayerAdd recibe dos activaciones; las demás, una, los parámetros van aparte
        const std::size_t input_count =
            node.operation == "LayerAdd" ? 2 : 1;

        if (node.inputs.size() != input_count) {
            throw std::invalid_argument(
                "Numero de entradas incorrecto en " + node.operation
            );
        }


        if (node.operation == "ReLU6" ||
            node.operation == "Tanh") {
            if (!node.parameters.empty()) {
                throw std::invalid_argument(
                    "La activacion no requiere parametros"
                );
            }

            const int size = required_int(node, "size");

            plans.push_back({
                node.operation == "ReLU6"
                    ? "relu6_forward"
                    : "tanh_forward",
                checked_tensor_bytes({size}),
                1
            });

        } else if (node.operation == "Linear") {
            if (node.parameters.size() != 2) {
                throw std::invalid_argument(
                    "Linear requiere pesos y bias"
                );
            }

            const int batch =
                required_int(node, "batch_size");

            const int in =
                required_int(node, "in_features");

            const int out =
                required_int(node, "out_features");

            required_int(node, "use_bias");

            // Comprueba que el tamaño de entrada puede representarse; aquí no mide el BO.
            checked_tensor_bytes({batch, in});

            const std::size_t weight_bytes =
                checked_tensor_bytes({out, in});

            const std::size_t bias_bytes =
                checked_tensor_bytes({out});

            // Este código exige capacidad para pesos y bias incluso si use_bias es falso.
            if (buffer(node.parameters[0]).size() < weight_bytes ||
                buffer(node.parameters[1]).size() < bias_bytes) {
                throw std::invalid_argument(
                    "Pesos o bias insuficientes para Linear"
                );
            }

            plans.push_back({
                "linear_forward",
                checked_tensor_bytes({batch, out}),
                3
            });

        } else if (node.operation == "Conv2D") {
            if (node.parameters.size() != 2) {
                throw std::invalid_argument(
                    "Conv2D requiere pesos y bias"
                );
            }

            const int b =
                required_int(node, "batch_size");

            const int ic =
                required_int(node, "input_channels");

            const int ih =
                required_int(node, "input_height");

            const int iw =
                required_int(node, "input_width");

            const int oc =
                required_int(node, "output_channels");

            const int oh =
                required_int(node, "output_height");

            const int ow =
                required_int(node, "output_width");

            const int kh =
                required_int(node, "kernel_height");

            const int kw =
                required_int(node, "kernel_width");

            const int sh =
                required_int(node, "stride_height");

            const int sw =
                required_int(node, "stride_width");

            const int ph =
                required_int(node, "padding_height");

            const int pw =
                required_int(node, "padding_width");

            required_int(node, "use_bias");

            if (b <= 0 || ic <= 0 || ih <= 0 || iw <= 0 ||
                oc <= 0 || oh <= 0 || ow <= 0 ||
                kh <= 0 || kw <= 0 ||
                sh <= 0 || sw <= 0 ||
                ph < 0 || pw < 0) {
                throw std::invalid_argument(
                    "Dimensiones o parametros invalidos en Conv2D"
                );
            }

            // Verifica salida = (entrada + 2 * padding - kernel) / stride + 1.
            // Usa enteros de 64 bits para los cálculos intermedios.
            const std::int64_t numerator_h =
                static_cast<std::int64_t>(ih) + 2LL * ph - kh;

            const std::int64_t numerator_w =
                static_cast<std::int64_t>(iw) + 2LL * pw - kw;

            if (numerator_h < 0 || numerator_w < 0 ||
                numerator_h / sh + 1 != oh ||
                numerator_w / sw + 1 != ow) {
                throw std::invalid_argument(
                    "Dimensiones de salida incorrectas en Conv2D"
                );
            }

            checked_tensor_bytes({b, ic, ih, iw});

            // Forma de los pesos: [canales de salida, canales de entrada, alto, ancho].
            const std::size_t weight_bytes =
                checked_tensor_bytes({oc, ic, kh, kw});

            const std::size_t bias_bytes =
                checked_tensor_bytes({oc});

            if (buffer(node.parameters[0]).size() < weight_bytes ||
                buffer(node.parameters[1]).size() < bias_bytes) {
                throw std::invalid_argument(
                    "Pesos o bias insuficientes para Conv2D"
                );
            }

            plans.push_back({
                "conv2d_forward",
                checked_tensor_bytes({b, oc, oh, ow}),
                3
            });

        } else if (node.operation == "AvgPool2D") {
            if (!node.parameters.empty()) {
                throw std::invalid_argument(
                    "AvgPool2D no requiere parametros"
                );
            }

            const int b = required_int(node, "batch_size");
            const int c = required_int(node, "input_channels");
            const int ih = required_int(node, "input_height");
            const int iw = required_int(node, "input_width");
            const int oh = required_int(node, "output_height");
            const int ow = required_int(node, "output_width");
            const int kh = required_int(node, "kernel_height");
            const int kw = required_int(node, "kernel_width");
            const int sh = required_int(node, "stride_height");
            const int sw = required_int(node, "stride_width");
            const int ph = required_int(node, "padding_height");
            const int pw = required_int(node, "padding_width");

            required_int(node, "count_include_pad");

            const bool use_override =
                required_int(node, "use_divisor_override") != 0;

            const int divisor =
                required_int(node, "divisor_override");

            if (b <= 0 || c <= 0 || ih <= 0 || iw <= 0 ||
                oh <= 0 || ow <= 0 || kh <= 0 || kw <= 0 ||
                sh <= 0 || sw <= 0 || ph < 0 || pw < 0) {
                throw std::invalid_argument(
                    "Dimensiones o parametros invalidos en AvgPool2D"
                );
            }

            if (use_override && divisor <= 0) {
                throw std::invalid_argument(
                    "divisor_override debe ser positivo cuando se utiliza"
                );
            }

            // Valida la forma de salida del pooling con división entera, sin redondear hacia arriba.
            const std::int64_t numerator_h =
                static_cast<std::int64_t>(ih) + 2LL * ph - kh;

            const std::int64_t numerator_w =
                static_cast<std::int64_t>(iw) + 2LL * pw - kw;

            if (numerator_h < 0 || numerator_w < 0 ||
                numerator_h / sh + 1 != oh ||
                numerator_w / sw + 1 != ow) {
                throw std::invalid_argument(
                    "Dimensiones de salida incorrectas en AvgPool2D"
                );
            }

            checked_tensor_bytes({b, c, ih, iw});

            plans.push_back({
                "avg_pool2d_forward",
                checked_tensor_bytes({b, c, oh, ow}),
                1
            });

        } else if (node.operation == "Pointwise") {
            if (node.parameters.size() != 2) {
                throw std::invalid_argument("Pointwise requiere pesos y bias");
            }

            const int b = required_int(node, "batch_size");
            const int ic = required_int(node, "input_channels");
            const int h = required_int(node, "input_height");
            const int w = required_int(node, "input_width");
            const int oc = required_int(node, "output_channels");
            required_int(node, "use_bias");

            checked_tensor_bytes({b, ic, h, w});
            const auto weight_bytes = checked_tensor_bytes({oc, ic});
            const auto bias_bytes = checked_tensor_bytes({oc});
            if (buffer(node.parameters[0]).size() < weight_bytes ||
                buffer(node.parameters[1]).size() < bias_bytes) {
                throw std::invalid_argument("Parametros insuficientes para Pointwise");
            }

            // Pointwise aplica convoluciones 1x1: cambia canales y conserva alto y ancho.
            plans.push_back({
                "pointwise_conv2d_forward",
                checked_tensor_bytes({b, oc, h, w}),
                3
            });

        } else if (node.operation == "Depthwise") {
            if (node.parameters.size() != 2) {
                throw std::invalid_argument("Depthwise requiere pesos y bias");
            }

            const int b = required_int(node, "batch_size");
            const int c = required_int(node, "channels");
            const int ih = required_int(node, "input_height");
            const int iw = required_int(node, "input_width");
            const int oh = required_int(node, "output_height");
            const int ow = required_int(node, "output_width");
            const int kh = required_int(node, "kernel_height");
            const int kw = required_int(node, "kernel_width");
            const int sh = required_int(node, "stride_height");
            const int sw = required_int(node, "stride_width");
            const int ph = required_int(node, "padding_height");
            const int pw = required_int(node, "padding_width");
            required_int(node, "use_bias");

            if (b <= 0 || c <= 0 || ih <= 0 || iw <= 0 ||
                oh <= 0 || ow <= 0 || kh <= 0 || kw <= 0 ||
                sh <= 0 || sw <= 0 || ph < 0 || pw < 0) {
                throw std::invalid_argument("Dimensiones invalidas en Depthwise");
            }

            const std::int64_t nh =
                static_cast<std::int64_t>(ih) + 2LL * ph - kh;
            const std::int64_t nw =
                static_cast<std::int64_t>(iw) + 2LL * pw - kw;
            if (nh < 0 || nw < 0 ||
                nh / sh + 1 != oh || nw / sw + 1 != ow) {
                throw std::invalid_argument("Salida incompatible con Depthwise");
            }

            checked_tensor_bytes({b, c, ih, iw});
            const auto weight_bytes = checked_tensor_bytes({c, kh, kw});
            const auto bias_bytes = checked_tensor_bytes({c});
            if (buffer(node.parameters[0]).size() < weight_bytes ||
                buffer(node.parameters[1]).size() < bias_bytes) {
                throw std::invalid_argument("Parametros insuficientes para Depthwise");
            }

            // Depthwise aplica un filtro espacial por canal y conserva la cantidad de canales.
            plans.push_back({
                "depthwise_conv2d_forward",
                checked_tensor_bytes({b, c, oh, ow}),
                3
            });

        } else if (node.operation == "BatchNorm2D") {
            if (node.parameters.size() != 4) {
                throw std::invalid_argument(
                    "BatchNorm2D requiere peso, bias, media y varianza"
                );
            }

            const int b = required_int(node, "batch_size");
            const int c = required_int(node, "channels");
            const int h = required_int(node, "input_height");
            const int w = required_int(node, "input_width");
            const float eps = required_float(node, "eps");
            required_int(node, "affine");
            required_int(node, "use_bias");
            if (eps <= 0.0f) {
                throw std::invalid_argument("eps debe ser positivo");
            }

            const auto bytes = checked_tensor_bytes({b, c, h, w});
            const auto param_bytes = checked_tensor_bytes({c});
            for (const auto& id : node.parameters) {
                if (buffer(id).size() < param_bytes) {
                    throw std::invalid_argument(
                        "Parametro insuficiente para BatchNorm2D: " + id
                    );
                }
            }

            // BatchNorm2D conserva la forma; sus cuatro parámetros tienen un valor por canal.
            plans.push_back({"batchnorm2d_forward", bytes, 5});

        } else if (node.operation == "LayerAdd") {
            if (!node.parameters.empty()) {
                throw std::invalid_argument("LayerAdd no requiere parametros");
            }

            const int size = required_int(node, "size");
            required_float(node, "alpha");

            // LayerAdd conserva ambas entradas hasta que termine el kernel que las suma.
            plans.push_back({
                "layer_add_forward", checked_tensor_bytes({size}), 2
            });
        } else {
            throw std::invalid_argument(
                "Operacion no admitida por el pipeline: " +
                node.operation
            );
        }

        // Cada salida necesita un uso posterior o la reserva de resultado final.
        if (graph_consumers(node.output) == 0) {
            throw std::invalid_argument(
                "Nodo sin consumidores ni salida del grafo: " +
                node.output
            );
        }

        producers.emplace(node.output, i);
    }

    // Comprueba que cada entrada de cada batch tenga un ID de BO distinto.
    std::unordered_set<std::string> unique_inputs;
    std::vector<std::string> input_ids;

    for (const auto& batch : batches) {
        // tensor_buffers relaciona el ID simbólico del grafo con el ID de BO del batch.
        // Al comenzar solo debe contener las entradas; las salidas se añaden al ejecutar.
        if (batch.tensor_buffers.size() != graph_inputs_.size()) {
            throw std::invalid_argument(
                "Cada batch debe contener unicamente las entradas del grafo"
            );
        }

        for (const auto& tensor : graph_inputs_) {
            const auto it = batch.tensor_buffers.find(tensor);

            if (it == batch.tensor_buffers.end() ||
                !has_buffer(it->second)) {
                throw std::invalid_argument(
                    "Falta el buffer de entrada: " + tensor
                );
            }

            const auto& id = it->second;

            if (!unique_inputs.insert(id).second) {
                throw std::invalid_argument(
                    "Los batches deben usar entradas distintas: " + id
                );
            }

            if (graph_parameters_.count(id) ||
                remaining_consumers_.count(id)) {
                throw std::logic_error(
                    "Entrada ocupada o con ciclo de vida previo: " + id
                );
            }

            if (graph_consumers(tensor) == 0) {
                throw std::invalid_argument(
                    "Entrada sin consumidores: " + tensor
                );
            }

            input_ids.push_back(id);
        }
    }

    // PipelineBatchState guarda el avance de cada batch. Sus estados son:
    // Pending, Queued (encolado), Running y Completed (finalización confirmada).
    for (std::size_t b = 0; b < batches.size(); ++b) {
        batches[b].batch_id = b;

        batches[b].node_states.assign(
            graph_nodes_.size(),
            PipelineNodeState::Pending
        );

        batches[b].completed_nodes = 0;
    }

    // Distingue los IDs de salida de esta ejecución frente a los de ejecuciones previas.
    const std::string prefix =
        "__pipeline_" +
        std::to_string(++pipeline_execution_id_) + "_";

    // Registra los IDs creados para poder retirarlos si ocurre un error.
    std::vector<std::string> created_ids;

    // Planificador anterior: un único hilo recoge los runs en orden de lanzamiento.
    std::vector<PipelineRunningOperation> running;
    std::unordered_set<std::string> used_kernels;
    for (const auto& plan : plans) used_kernels.insert(plan.kernel_name);
    running.reserve(used_kernels.size());

    graph_running_ = true;

    try {
        // Aplica a cada BO de entrada los usos calculados para su tensor en el grafo.
        for (const auto& batch : batches) {
            for (const auto& tensor : graph_inputs_) {
                configure_lifetime(
                    batch.tensor_buffers.at(tensor),
                    graph_consumers(tensor)
                );
            }
        }

        // La ejecución termina cuando todos los batches completaron todos sus nodos.
        const auto all_completed = [&]() {
            return std::all_of(
                batches.begin(),
                batches.end(),
                [&](const PipelineBatchState& batch) {
                    return batch.completed_nodes ==
                           graph_nodes_.size();
                }
            );
        };

        while (!all_completed()) {
            for (std::size_t b = 0; b < batches.size(); ++b) {
                auto& batch = batches[b];
                for (const auto index : graph_order_) {
                    const auto& node = graph_nodes_[index];
                    const auto& plan = plans[index];
                    if (batch.node_states[index] != PipelineNodeState::Pending)
                        continue;
                    bool ready = true;
                    for (const auto& input : node.inputs) {
                        const auto producer = producers.find(input);
                        if (producer != producers.end() &&
                            batch.node_states[producer->second] != PipelineNodeState::Completed) {
                            ready = false;
                            break;
                        }
                    }
                    if (!ready) continue;
                    const bool busy = std::any_of(
                        running.begin(), running.end(),
                        [&](const PipelineRunningOperation& active) {
                            return active.kernel_name == plan.kernel_name;
                        });
                    if (busy) continue;
                    // Crea un ID propio para esta combinación de ejecución, batch y nodo.
                    const std::string output_id =
                        prefix + std::to_string(b) + "_" +
                        std::to_string(index);

                    if (has_buffer(output_id) ||
                        unique_inputs.count(output_id) ||
                        graph_parameters_.count(output_id)) {
                        throw std::logic_error(
                            "Identificador de pipeline ocupado: " +
                            output_id
                        );
                    }

                    created_ids.push_back(output_id);

                    auto& output_bo = create_buffer(
                        output_id,
                        plan.output_bytes,
                        get_kernel(plan.kernel_name).group_id(
                            plan.output_argument
                        )
                    );

                    configure_lifetime(
                        output_id,
                        graph_consumers(node.output)
                    );

                    // Asocia la salida simbólica con el BO específico de este batch.
                    batch.tensor_buffers.emplace(
                        node.output,
                        output_id
                    );

                    auto& input_bo = buffer(
                        batch.tensor_buffers.at(node.inputs[0])
                    );

                    // Prepara el registro que permitirá identificar y esperar este trabajo.
                    PipelineRunningOperation active{
                        b,
                        index,
                        plan.kernel_name,
                        xrt::run(get_kernel(plan.kernel_name))
                    };

                    // Los wrappers async lanzan el kernel y devuelven su run sin esperar.
                    // Así pueden enviarse trabajos a otros kernels antes de recoger este resultado.
                    // En este bloque se seleccionan los wrappers

                    if (node.operation == "ReLU6") {
                        active.run = relu6_fpga_forward_async(
                            input_bo,
                            output_bo,
                            required_int(node, "size")
                        );

                    } else if (node.operation == "Tanh") {
                        active.run = tanh_fpga_forward_async(
                            input_bo,
                            output_bo,
                            required_int(node, "size")
                        );

                    } else if (node.operation == "Linear") {
                        active.run = linear_fpga_forward_async(
                            input_bo,
                            buffer(node.parameters[0]),
                            buffer(node.parameters[1]),
                            output_bo,
                            required_int(node, "batch_size"),
                            required_int(node, "in_features"),
                            required_int(node, "out_features"),
                            required_int(node, "use_bias") != 0
                        );

                    } else if (node.operation == "Conv2D") {
                        active.run = conv2d_fpga_forward_async(
                            input_bo,
                            buffer(node.parameters[0]),
                            buffer(node.parameters[1]),
                            output_bo,
                            required_int(node, "batch_size"),
                            required_int(node, "input_channels"),
                            required_int(node, "input_height"),
                            required_int(node, "input_width"),
                            required_int(node, "output_channels"),
                            required_int(node, "output_height"),
                            required_int(node, "output_width"),
                            required_int(node, "kernel_height"),
                            required_int(node, "kernel_width"),
                            required_int(node, "stride_height"),
                            required_int(node, "stride_width"),
                            required_int(node, "padding_height"),
                            required_int(node, "padding_width"),
                            required_int(node, "use_bias") != 0
                        );

                    } else if (node.operation == "AvgPool2D") {
                        active.run = avgpool2d_fpga_forward_async(
                            input_bo,
                            output_bo,
                            required_int(node, "batch_size"),
                            required_int(node, "input_channels"),
                            required_int(node, "input_height"),
                            required_int(node, "input_width"),
                            required_int(node, "output_height"),
                            required_int(node, "output_width"),
                            required_int(node, "kernel_height"),
                            required_int(node, "kernel_width"),
                            required_int(node, "stride_height"),
                            required_int(node, "stride_width"),
                            required_int(node, "padding_height"),
                            required_int(node, "padding_width"),
                            required_int(node, "count_include_pad") != 0,
                            required_int(node, "use_divisor_override") != 0,
                            required_int(node, "divisor_override")
                        );

                    } else if (node.operation == "Pointwise") {
                        active.run = pointwise_fpga_forward_async(
                            input_bo,
                            buffer(node.parameters[0]),
                            buffer(node.parameters[1]),
                            output_bo,
                            required_int(node, "batch_size"),
                            required_int(node, "input_channels"),
                            required_int(node, "input_height"),
                            required_int(node, "input_width"),
                            required_int(node, "output_channels"),
                            required_int(node, "use_bias") != 0
                        );

                    } else if (node.operation == "Depthwise") {
                        active.run = depthwise_fpga_forward_async(
                            input_bo,
                            buffer(node.parameters[0]),
                            buffer(node.parameters[1]),
                            output_bo,
                            required_int(node, "batch_size"),
                            required_int(node, "channels"),
                            required_int(node, "input_height"),
                            required_int(node, "input_width"),
                            required_int(node, "output_height"),
                            required_int(node, "output_width"),
                            required_int(node, "kernel_height"),
                            required_int(node, "kernel_width"),
                            required_int(node, "stride_height"),
                            required_int(node, "stride_width"),
                            required_int(node, "padding_height"),
                            required_int(node, "padding_width"),
                            required_int(node, "use_bias") != 0
                        );

                    } else if (node.operation == "BatchNorm2D") {
                        // Parámetros persistentes en orden: peso, bias, media y varianza.
                        active.run = batchnorm2d_fpga_forward_async(
                            input_bo,
                            buffer(node.parameters[0]),
                            buffer(node.parameters[1]),
                            buffer(node.parameters[2]),
                            buffer(node.parameters[3]),
                            output_bo,
                            required_int(node, "batch_size"),
                            required_int(node, "channels"),
                            required_int(node, "input_height"),
                            required_int(node, "input_width"),
                            required_float(node, "eps"),
                            required_int(node, "affine") != 0,
                            required_int(node, "use_bias") != 0
                        );

                    } else if (node.operation == "LayerAdd") {
                        // La segunda entrada de LayerAdd también pertenece al batch actual.
                        auto& input2_bo = buffer(
                            batch.tensor_buffers.at(node.inputs[1])
                        );
                        active.run = layeradd_fpga_forward_async(
                            input_bo,
                            input2_bo,
                            output_bo,
                            required_int(node, "size"),
                            required_float(node, "alpha")
                        );
                    }

                    running.push_back(std::move(active));
                    batch.node_states[index] = PipelineNodeState::Running;
                }
            }
            if (running.empty())
                throw std::logic_error("Hay nodos pendientes pero ninguno puede avanzar");

            // Espera el trabajo más antiguo, aunque otro kernel haya terminado antes.
            auto& active = running.front();
            if (active.run.wait() != ERT_CMD_STATE_COMPLETED)
                throw std::runtime_error("Fallo el kernel: " + active.kernel_name);
            auto& batch = batches[active.batch_index];
            const auto& node = graph_nodes_[active.node_index];
            for (const auto& input : node.inputs)
                consume_buffer(batch.tensor_buffers.at(input));
            batch.node_states[active.node_index] = PipelineNodeState::Completed;
            ++batch.completed_nodes;
            running.erase(running.begin());
        }

        // Las salidas finales siguen reservadas: quien llama debe descargarlas
        // y después consumir sus BO usando los IDs guardados en tensor_buffers.
        graph_running_ = false;

    } catch (...) {
        // Espera los trabajos registrados antes de descartar sus buffers.
        // Si wait() lanza, se conservan los recursos y el grafo permanece activo.
        for (auto& active : running) active.run.wait();

        // Retira un BO y sus registros sin enviarlo al pool de reutilización.
        const auto discard = [&](const std::string& id) {
            buffers_.erase(id);
            memory_groups_.erase(id);
            remaining_consumers_.erase(id);
            persistent_.erase(id);
        };

        // Descarta las entradas y salidas de esta ejecución; conserva los parámetros.
        for (const auto& id : input_ids) {
            discard(id);
        }

        for (const auto& id : created_ids) {
            discard(id);
        }

        // Descarta los BO libres y vacía el estado de los batches antes de propagar el error.
        pool_.clear();

        for (auto& batch : batches) {
            batch.tensor_buffers.clear();
            batch.node_states.clear();
            batch.completed_nodes = 0;
        }

        graph_running_ = false;
        throw;
    }
}

// ------------------------------------------------------------
// 7. Reutilización de kernels y cierre de recursos
// ------------------------------------------------------------
// Devuelve el kernel guardado o lo abre si es la primera solicitud de ese nombre.
xrt::kernel& FpgaRuntime::get_kernel(const std::string& name)
{
    // try_emplace construye el objeto solo si la clave no existe.
    // it apunta a la entrada; inserted indica si se añadió un kernel nuevo.
    auto [it, inserted] = kernels_.try_emplace(
        name,
        device_,
        uuid_,
        name
    );

    return it->second;
}

// Vacía los BO y kernels administrados por el runtime cuando el grafo está inactivo.
void FpgaRuntime::shutdown()
{
    if (graph_running_)
        throw std::logic_error("No se puede liberar XRT durante una ejecución");

    std::cout << "[XRT] Liberando buffers..." << std::endl;

    buffers_.clear();
    pool_.clear();

    remaining_consumers_.clear();
    persistent_.clear();
    memory_groups_.clear();

    std::cout << "[XRT] Liberando kernels..." << std::endl;

    kernels_.clear();

    std::cout << "[XRT] Recursos liberados correctamente" << std::endl;
}

