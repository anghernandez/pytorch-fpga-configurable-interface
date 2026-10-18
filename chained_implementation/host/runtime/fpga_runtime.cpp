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

// ============================================================
// 1. INSTANCIA COMPARTIDA E INICIALIZACIÓN DE XRT
// ============================================================
FpgaRuntime& FpgaRuntime::instance()
{
    static FpgaRuntime instance;
    return instance;
}

FpgaRuntime::FpgaRuntime() : device_(0)
{
    const std::string xclbin_path =
        "/lib/firmware/xilinx/fpga_accelerators/fpga_kernels.xclbin";
    std::cout << "[FpgaRuntime] Cargando: " << xclbin_path << std::endl;
    uuid_ = device_.load_xclbin(xclbin_path);
}

xrt::device& FpgaRuntime::device() { return device_; }
const xrt::uuid& FpgaRuntime::uuid() const { return uuid_; }

// ============================================================
// 2. RESERVA O REUTILIZACIÓN DE UN BUFFER
// ============================================================
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

    // Reutilizar solamente BO del mismo grupo y con capacidad suficiente.
    // No se asume que todos los grupos HP son intercambiables.
    for (auto it = pool_.begin(); it != pool_.end(); ++it) {
        if (it->first == memory_group && it->second.size() >= bytes) {
            buffers_.emplace(id, std::move(it->second));
            pool_.erase(it);
            memory_groups_.emplace(id, memory_group);
            ++reuse_count_;
            return buffers_.at(id);
        }
    }

    auto result = buffers_.try_emplace(id, device_, bytes, memory_group);
    memory_groups_.emplace(id, memory_group);
    return result.first->second;
}

// ============================================================
// 2.1. CARGA ÚNICA DE PARÁMETROS PERSISTENTES
// ============================================================
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
    if (has_buffer(id))
        throw std::invalid_argument("El parámetro ya existe: " + id);

    xrt::bo& bo = create_buffer(id, count * sizeof(float), memory_group);
    try {
        std::copy_n(values, count, bo.map<float*>());
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        configure_lifetime(id, 0, true);
    } catch (...) {
        // No dejar un identificador activo tras una carga incompleta.
        release_buffer(id);
        throw;
    }
    return bo;
}

xrt::bo& FpgaRuntime::buffer(const std::string& id)
{
    auto it = buffers_.find(id);
    if (it == buffers_.end())
        throw std::out_of_range("Buffer no encontrado: " + id);
    return it->second;
}

bool FpgaRuntime::has_buffer(const std::string& id) const
{
    return buffers_.find(id) != buffers_.end();
}

// ============================================================
// 3. REGISTRO EXPLÍCITO DE DEPENDENCIAS Y PERMANENCIA
// ============================================================
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

// ============================================================
// 4. RECICLAJE DE MEMORIA Y CONSULTAS
// ============================================================
void FpgaRuntime::release_buffer(const std::string& id)
{
    auto it = buffers_.find(id);
    if (it == buffers_.end())
        throw std::out_of_range("Buffer no encontrado: " + id);

    auto pending = remaining_consumers_.find(id);
    if (pending != remaining_consumers_.end() && pending->second != 0)
        throw std::logic_error("Buffer con consumidores pendientes: " + id);

    // Los wrappers ejecutan run.wait() antes de que se informe el consumo.
    // Nunca llamar a release_buffer mientras el kernel todavía esté en curso.
    const int group = memory_groups_.at(id);
    pool_.emplace_back(group, std::move(it->second));
    buffers_.erase(it);
    memory_groups_.erase(id);
    remaining_consumers_.erase(id);
    persistent_.erase(id);
}

std::size_t FpgaRuntime::buffer_count() const { return buffers_.size(); }
std::size_t FpgaRuntime::pooled_buffer_count() const { return pool_.size(); }
std::size_t FpgaRuntime::reuse_count() const { return reuse_count_; }
void FpgaRuntime::clear_pool() { pool_.clear(); }

// ============================================================
// 5. REGISTRO Y VALIDACIÓN DE UN GRAFO DE EJECUCIÓN DIFERIDA
// ============================================================
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

void FpgaRuntime::add_graph_input(const std::string& id)
{
    if (graph_finalized_)
        throw std::logic_error("El grafo ya fue finalizado");
    if (id.empty() || graph_parameters_.count(id) || !graph_inputs_.insert(id).second)
        throw std::invalid_argument("Entrada de grafo inválida o repetida: " + id);
}

void FpgaRuntime::add_graph_parameter(const std::string& id)
{
    if (graph_finalized_)
        throw std::logic_error("El grafo ya fue finalizado");
    if (id.empty() || graph_inputs_.count(id) || !graph_parameters_.insert(id).second)
        throw std::invalid_argument("Parámetro de grafo inválido o repetido: " + id);
}

void FpgaRuntime::add_operation(const OperationNode& node)
{
    if (graph_finalized_)
        throw std::logic_error("El grafo ya fue finalizado");
    if (node.operation.empty() || node.output.empty() || node.inputs.empty())
        throw std::invalid_argument("La operación necesita nombre, entradas y salida");
    graph_nodes_.push_back(node);
}

void FpgaRuntime::set_graph_outputs(const std::vector<std::string>& ids)
{
    if (graph_finalized_)
        throw std::logic_error("El grafo ya fue finalizado");
    if (ids.empty())
        throw std::invalid_argument("El grafo necesita al menos una salida");
    graph_outputs_ = ids;
}

void FpgaRuntime::finalize_graph()
{
    if (graph_finalized_)
        throw std::logic_error("El grafo ya fue finalizado");
    if (graph_outputs_.empty())
        throw std::logic_error("Primero configure las salidas del grafo");

    // Se calcula primero en variables locales: un error no deja el grafo a medias.
    std::unordered_map<std::string, std::size_t> producer;
    for (std::size_t i = 0; i < graph_nodes_.size(); ++i) {
        const auto& id = graph_nodes_[i].output;
        if (graph_inputs_.count(id) || graph_parameters_.count(id) ||
            !producer.emplace(id, i).second)
            throw std::invalid_argument("Tensor producido más de una vez: " + id);
    }

    std::vector<std::vector<std::size_t>> dependents(graph_nodes_.size());
    std::vector<std::size_t> indegree(graph_nodes_.size(), 0);
    std::unordered_map<std::string, std::size_t> consumers;

    for (std::size_t i = 0; i < graph_nodes_.size(); ++i) {
        const auto& node = graph_nodes_[i];
        for (const auto& input : node.inputs) {
            if (!graph_inputs_.count(input) && !producer.count(input))
                throw std::invalid_argument("Entrada sin productor: " + input);
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
        // Las salidas se retienen para recuperar sus datos después del grafo.
        ++consumers[id];
    }

    std::queue<std::size_t> ready;
    for (std::size_t i = 0; i < indegree.size(); ++i)
        if (indegree[i] == 0) ready.push(i);

    std::vector<std::size_t> order;
    while (!ready.empty()) {
        const auto i = ready.front();
        ready.pop();
        order.push_back(i);
        for (const auto successor : dependents[i])
            if (--indegree[successor] == 0) ready.push(successor);
    }
    if (order.size() != graph_nodes_.size())
        throw std::invalid_argument("El grafo contiene un ciclo");

    graph_order_ = std::move(order);
    graph_consumers_ = std::move(consumers);
    graph_finalized_ = true;
}

const std::vector<OperationNode>& FpgaRuntime::graph_nodes() const
{
    return graph_nodes_;
}

const std::vector<std::size_t>& FpgaRuntime::graph_execution_order() const
{
    if (!graph_finalized_)
        throw std::logic_error("El grafo no fue finalizado");
    return graph_order_;
}

std::size_t FpgaRuntime::graph_consumers(const std::string& id) const
{
    if (!graph_finalized_)
        throw std::logic_error("El grafo no fue finalizado");
    const auto it = graph_consumers_.find(id);
    return it == graph_consumers_.end() ? 0 : it->second;
}

// ============================================================
// 6. VINCULACIÓN DEL DAG CON EL CICLO DE VIDA DE LOS BUFFERS
// ============================================================
void FpgaRuntime::begin_graph_execution()
{
    if (!graph_finalized_ || graph_running_)
        throw std::logic_error("El DAG debe estar finalizado e inactivo");

    // Evitar iniciar con referencias caducadas o reutilizadas accidentalmente.
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

    // Los datos persistentes no se reciclan cuando termina un consumidor.
    for (const auto& id : graph_parameters_)
        if (!remaining_consumers_.count(id))
            configure_lifetime(id, 0, true);
    for (const auto& id : graph_inputs_)
        configure_lifetime(id, graph_consumers(id), graph_consumers(id) == 0);

    graph_next_node_ = 0;
    graph_running_ = true;
}

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

    // Cada aparición de una entrada cuenta como un consumo; necesario
    // si una operación recibe el mismo tensor en varios argumentos.
    for (const auto& id : node.inputs)
        consume_buffer(id);
    ++graph_next_node_;
}

void FpgaRuntime::end_graph_execution()
{
    if (!graph_running_ || graph_next_node_ != graph_order_.size())
        throw std::logic_error("La ejecución del DAG está incompleta");
    graph_running_ = false;
    graph_next_node_ = 0;
    // Las salidas del grafo mantienen una reserva y deben recuperarse
    // antes de llamar consume_buffer(id) para su liberación.
}

// ============================================================
// 7. DESPACHADOR DEL DAG: OPERACIONES INDEPENDIENTES DEL MODELO
// ============================================================
void FpgaRuntime::execute_graph()
{
    if (!graph_finalized_ || graph_running_)
        throw std::logic_error("El DAG debe estar finalizado e inactivo");

    // Validar la descripción antes de ejecutar el primer kernel.
    // Los argumentos son valores por nodo, NO propiedades del modelo.
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
    // Wrappers sincrónicos: cada llamada espera run.wait().
    // Si falla un kernel, no se marca su nodo como completado.
    for (const auto index : graph_order_) {
        const auto& node = graph_nodes_.at(index);
        const auto& name = node.operation;
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
        const auto input = [&](std::size_t i) -> xrt::bo& {
            return buffer(node.inputs.at(i));
        };
        const auto parameter = [&](std::size_t i) -> xrt::bo& {
            return buffer(node.parameters.at(i));
        };
        // Un tensor de salida siempre es float32 en los kernels actuales.
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
            xrt::kernel kernel(device_, uuid_, kernel_name);
            return create_graph_output_buffer(node.output,
                                              count * sizeof(float),
                                              kernel.group_id(output_index));
        };

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
        complete_graph_operation(index);
    }
    end_graph_execution();
    } catch (...) {
        // Recuperación: no conservar salidas parciales ni contadores
        // de una inferencia interrumpida. Los parámetros persistentes
        // no se modifican y el DAG sigue disponible para otro intento.
        // Se descartan los BO afectados (sin reciclarlos) por seguridad.
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
        // El pool puede contener BO reciclados por nodos completados.
        // Tras una excepción no se reutilizan hasta una nueva asignación.
        pool_.clear();
        graph_running_ = false;
        graph_next_node_ = 0;
        throw;
    }
}