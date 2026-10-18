#ifndef FPGA_RUNTIME_HPP
#define FPGA_RUNTIME_HPP

#include <cstddef>
#include <unordered_set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_uuid.h>

// Un nodo describe una operación; todavía NO ejecuta el kernel.
struct OperationNode {
    std::string operation;
    std::vector<std::string> inputs;   // Tensores de activación (no pesos)
    std::vector<std::string> parameters; // IDs de parámetros persistentes
    std::string output;
    std::unordered_map<std::string, int> int_args;
    std::unordered_map<std::string, float> float_args;
};

class FpgaRuntime
{
public:
    static FpgaRuntime& instance();

    xrt::device& device();
    const xrt::uuid& uuid() const;

    // 1. Reservar o reutilizar memoria para un identificador nuevo.
    xrt::bo& create_buffer(const std::string& id,
                           std::size_t bytes,
                           int memory_group);
    // Cargar una vez parámetros float32 y conservarlos entre inferencias.
    // Los IDs persistentes no deben coincidir con entradas ni salidas del grafo.
    xrt::bo& load_parameter(const std::string& id,
                            const float* values,
                            std::size_t count,
                            int memory_group);
    xrt::bo& buffer(const std::string& id);
    bool has_buffer(const std::string& id) const;

    // 2. Registrar consumidores FUTUROS (incluidas ramas residuales).
    // Se configura antes de ejecutar las operaciones que consumen el buffer.
    // persistent=true: parámetros o datos conservados hasta liberación manual.
    void configure_lifetime(const std::string& id,
                            std::size_t consumers,
                            bool persistent = false);

    // 3. Notificar el fin de un consumo, DESPUÉS de run.wait().
    // Cuando queda cero y no es persistente, devuelve el BO al conjunto reutilizable.
    void consume_buffer(const std::string& id);

    // 4. Liberación explícita; rechaza buffers con consumidores pendientes.
    void release_buffer(const std::string& id);
    std::size_t buffer_count() const;
    std::size_t pooled_buffer_count() const;
    std::size_t reuse_count() const;
    void clear_pool();

    // 5. Construir un DAG antes de ejecutar los kernels.
    void clear_graph();
    void add_graph_input(const std::string& tensor_id);
    void add_graph_parameter(const std::string& parameter_id);
    void add_operation(const OperationNode& node);
    void set_graph_outputs(const std::vector<std::string>& tensor_ids);
    void finalize_graph();
    const std::vector<OperationNode>& graph_nodes() const;
    const std::vector<std::size_t>& graph_execution_order() const;
    std::size_t graph_consumers(const std::string& tensor_id) const;

    // 6. Conectar el DAG finalizado con el ciclo de vida de los BO.
    // Las entradas/parámetros deben estar cargados antes de comenzar.
    void begin_graph_execution();
    xrt::bo& create_graph_output_buffer(const std::string& id,
                                       std::size_t bytes,
                                       int memory_group);
    // Se llama únicamente DESPUÉS de ejecutar y esperar el kernel del nodo.
    void complete_graph_operation(std::size_t node_index);
    void end_graph_execution();

    // Ejecuta el DAG finalizado completo, usando los wrappers C++.
    // Entradas y parámetros deben estar cargados en BO previamente.
    void execute_graph();


private:
    FpgaRuntime();
    FpgaRuntime(const FpgaRuntime&) = delete;
    FpgaRuntime& operator=(const FpgaRuntime&) = delete;

    xrt::device device_;
    xrt::uuid uuid_;

    // Los BO activos son propiedad del runtime y accesibles mediante id.
    std::unordered_map<std::string, xrt::bo> buffers_;
    std::unordered_map<std::string, int> memory_groups_;
    std::unordered_map<std::string, std::size_t> remaining_consumers_;
    std::unordered_map<std::string, bool> persistent_;

    // BO libres indexados por grupo de memoria; se elige uno con tamaño >= pedido.
    std::vector<std::pair<int, xrt::bo>> pool_;
    std::size_t reuse_count_ = 0;

    // Definición simbólica del grafo, separada de los BO existentes.
    std::vector<OperationNode> graph_nodes_;
    std::unordered_set<std::string> graph_inputs_;
    std::unordered_set<std::string> graph_parameters_;
    std::vector<std::string> graph_outputs_;
    std::vector<std::size_t> graph_order_;
    std::unordered_map<std::string, std::size_t> graph_consumers_;
    bool graph_finalized_ = false;
    bool graph_running_ = false;
    std::size_t graph_next_node_ = 0;
};

#endif // FPGA_RUNTIME_HPP