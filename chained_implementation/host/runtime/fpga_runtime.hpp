#ifndef FPGA_RUNTIME_HPP
#define FPGA_RUNTIME_HPP

#include <cstddef>        // Define std::size_t para tamaños e índices
#include <unordered_set> // Conjunto de elementos únicos
#include <string>        // Manejo de cadenas de texto
#include <unordered_map> // Asociación de claves con valores
#include <utility>       // Permite utilizar std::pair
#include <vector>        // Contenedores dinámicos

#include <xrt/xrt/xrt_bo.h>     // Gestión de buffers XRT
#include <xrt/xrt/xrt_device.h> // Acceso al dispositivo FPGA
#include <xrt/xrt/xrt_uuid.h>   // Identificación del xclbin cargado
#include <xrt/xrt/xrt_kernel.h> // Control y ejecución de kernels XRT
#include <cstdint>             // Tipos enteros de tamaño definido


// ------------------------------------------------------------
// Representa simbólicamente una operación dentro del DAG.
// Almacena el tipo de operación, entradas, parámetros,
// salida y argumentos necesarios para ejecutar el kernel
// Su creación no implica ejecutar una operación en la FPGA
// ------------------------------------------------------------
struct OperationNode {
    std::string operation; // Nombre de la operación.
    std::vector<std::string> inputs; // IDs de tensores de entrada.
    std::vector<std::string> parameters; // IDs de parámetros persistentes.
    std::string output; // ID del tensor de salida.

    // Argumentos de configuración asociados a la operación.
    std::unordered_map<std::string, int> int_args;
    std::unordered_map<std::string, float> float_args;
};


// ------------------------------------------------------------
// Define los posibles estados de ejecución de un nodo
// del grafo para un batch determinado.
// Permite controlar qué operaciones están pendientes,
// ejecutándose o finalizadas.
// ------------------------------------------------------------
enum class PipelineNodeState {
    Pending,   // Todavía no se ha encolado.
    Queued,    // En la cola del kernel; dependencias completadas.
    Running,   // El kernel está en ejecución.
    Completed  // El kernel terminó correctamente.
};

// ------------------------------------------------------------
// Almacena el estado de ejecución de un batch en el pipeline
// Cada batch mantiene sus propios buffers de activación
// y estados de los nodos, mientras que los parámetros
// del modelo pueden compartirse entre batches
// ------------------------------------------------------------
struct PipelineBatchState {
    std::size_t batch_id = 0;   // Identificador del batch.

    // Relaciona los IDs simbólicos de los tensores con los IDs
    // de los buffers XRT asignados al batch
    std::unordered_map<std::string, std::string> tensor_buffers;

    // Estado de ejecución de cada nodo del grafo.
    std::vector<PipelineNodeState> node_states;

    // Cantidad de nodos que han terminado su ejecución.
    std::size_t completed_nodes = 0;
};

// ------------------------------------------------------------
// Representa una operación del pipeline enviada a XRT.
// Permite identificar el batch, el nodo y el kernel
// asociados a una ejecución todavía en seguimiento.
// ------------------------------------------------------------
struct PipelineRunningOperation {
    std::size_t batch_index; // Índice del batch en ejecución
    std::size_t node_index; // Índice del nodo dentro del DAG
    std::string kernel_name; // Nombre del kernel FPGA
    xrt::run run; // Objeto XRT para controlar la ejecución
};


// ------------------------------------------------------------
// Almacena estadísticas relacionadas con la gestión
// y reutilización de buffers XRT
// Permite registrar creaciones, reutilizaciones,
// causas de rechazo y tiempos de gestión
// ------------------------------------------------------------
struct BufferProfile {
    std::size_t created = 0;
    std::size_t reused = 0;

    std::size_t pool_empty = 0;
    std::size_t group_mismatch = 0;
    std::size_t insufficient_capacity = 0;

    double creation_ms = 0.0;
    double reuse_ms = 0.0;
};

// ------------------------------------------------------------
// Clase principal para administrar la ejecución FPGA
// Gestiona el dispositivo XRT, los kernels, los buffers,
// la representación del DAG y su ejecución
// También proporciona soporte para procesar múltiples
// batches mediante un planificador de pipeline
// ------------------------------------------------------------
class FpgaRuntime
{
public:

    // ------------------------------------------------------------
    // Proporciona acceso a una instancia compartida del runtime
    // Permite utilizar el mismo dispositivo, kernels y buffers
    // desde diferentes componentes de la implementación
    // ------------------------------------------------------------
    static FpgaRuntime& instance();

    xrt::device& device();
    const xrt::uuid& uuid() const;
    

    // ------------------------------------------------------------
    // Administra los buffers XRT identificados mediante IDs
    // Permite crear, recuperar, reutilizar y liberar memoria,
    // considerando su capacidad, grupo de memoria y ciclo de vida
    // ------------------------------------------------------------
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

    // Registra la cantidad de consumidores pendientes de un buffer
    // y si debe conservarse de manera persistente.
    void configure_lifetime(const std::string& id,
                            std::size_t consumers,
                            bool persistent = false);

    // Actualiza los consumidores pendientes de un buffer
    // Cuando ya no tiene consumidores y no es persistente,
    // puede devolverse al conjunto de buffers reutilizables
    void consume_buffer(const std::string& id);

    // Permite liberar explícitamente un buffer que no tenga
    // consumidores pendientes
    void release_buffer(const std::string& id);
    std::size_t buffer_count() const;
    std::size_t pooled_buffer_count() const;
    std::size_t reuse_count() const;
    void clear_pool();

    
    // ------------------------------------------------------------
    // Funciones para construir y administrar el DAG.
    // Permiten registrar entradas, parámetros, operaciones
    // y salidas, además de finalizar el grafo y consultar
    // su orden de ejecución y dependencias.
    // ------------------------------------------------------------
    void clear_graph();
    void add_graph_input(const std::string& tensor_id);
    void add_graph_parameter(const std::string& parameter_id);
    void add_operation(const OperationNode& node);
    void set_graph_outputs(const std::vector<std::string>& tensor_ids);
    void finalize_graph();
    const std::vector<OperationNode>& graph_nodes() const;
    const std::vector<std::size_t>& graph_execution_order() const;
    std::size_t graph_consumers(const std::string& tensor_id) const;

    // ------------------------------------------------------------
    // Funciones para controlar el ciclo de ejecución del DAG.
    // Relacionan las operaciones con los buffers XRT y permiten
    // administrar el consumo y liberación de memoria conforme
    // finalizan los nodos del grafo.
    // ------------------------------------------------------------
    void begin_graph_execution();
    xrt::bo& create_graph_output_buffer(const std::string& id,
                                       std::size_t bytes,
                                       int memory_group);
    // Se llama únicamente DESPUÉS de ejecutar y esperar el kernel del nodo.
    void complete_graph_operation(std::size_t node_index);
    void end_graph_execution();

    // ------------------------------------------------------------
    // Ejecuta las operaciones del DAG finalizado.
    // Utiliza los wrappers C++ y respeta las dependencias
    // entre nodos mediante una ejecución secuencial.
    // ------------------------------------------------------------
    void execute_graph();

    // ------------------------------------------------------------
    // Ejecuta múltiples batches utilizando el mismo DAG.
    // Administra los estados de cada batch y permite coordinar
    // operaciones de distintos batches para aprovechar
    // la concurrencia entre kernels FPGA.
    // ------------------------------------------------------------
    void execute_pipeline(std::vector<PipelineBatchState>& batches);

    // Recupera un kernel XRT mediante su nombre
    xrt::kernel& get_kernel(const std::string& name);
    
    // Permite realizar el cierre y limpieza del runtime.
    void shutdown();

private:

    // ------------------------------------------------------------
    // Constructor privado para controlar la creación del runtime.
    // Se deshabilitan las operaciones de copia para impedir
    // la duplicación accidental de la instancia.
    // ------------------------------------------------------------
    FpgaRuntime();
    FpgaRuntime(const FpgaRuntime&) = delete;
    FpgaRuntime& operator=(const FpgaRuntime&) = delete;

    // ------------------------------------------------------------
    // Recursos asociados al dispositivo FPGA
    // Mantienen la referencia al dispositivo XRT y al UUID
    // correspondiente al xclbin cargado
    // ------------------------------------------------------------
    xrt::device device_;
    xrt::uuid uuid_;


    // Almacena los kernels XRT identificados por nombre
    // para permitir su recuperación y reutilización.
    std::unordered_map<std::string, xrt::kernel> kernels_;

    // ------------------------------------------------------------
    // Estructuras para administrar los buffers XRT.
    // Registran buffers activos, grupos de memoria,
    // consumidores pendientes y persistencia.
    // ------------------------------------------------------------
    std::unordered_map<std::string, xrt::bo> buffers_;
    std::unordered_map<std::string, int> memory_groups_;
    std::unordered_map<std::string, std::size_t> remaining_consumers_;
    std::unordered_map<std::string, bool> persistent_;

    // Almacena buffers disponibles para reutilización,
    // asociados con sus respectivos grupos de memoria
    std::vector<std::pair<int, xrt::bo>> pool_;

    // Contabiliza las reutilizaciones de buffers.
    std::size_t reuse_count_ = 0;
    
    // Identifica las ejecuciones del pipeline para evitar
    // colisiones entre los nombres de buffers de distintos ciclos
    std::size_t pipeline_execution_id_ = 0;

    // ------------------------------------------------------------
    // Estructuras internas que almacenan la definición del DAG.
    // Registran nodos, entradas, parámetros, salidas,
    // orden de ejecución y cantidad de consumidores.
    // ------------------------------------------------------------
    std::vector<OperationNode> graph_nodes_;
    std::unordered_set<std::string> graph_inputs_;
    std::unordered_set<std::string> graph_parameters_;
    std::vector<std::string> graph_outputs_;
    std::vector<std::size_t> graph_order_;
    std::unordered_map<std::string, std::size_t> graph_consumers_;

    // ------------------------------------------------------------
    // Variables de control del DAG.
    // Indican si el grafo está finalizado, si se encuentra
    // en ejecución y cuál es el siguiente nodo por procesar.
    // ------------------------------------------------------------
    bool graph_finalized_ = false;
    bool graph_running_ = false;
    std::size_t graph_next_node_ = 0;

};

#endif // FPGA_RUNTIME_HPP


