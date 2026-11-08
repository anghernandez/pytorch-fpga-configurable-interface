#include "linear_fpga.hpp"  // Declaración de las funciones del wrapper.
#include "fpga_runtime.hpp" // Acceso al runtime y kernels FPGA.

#include <cstddef>          // Define std::size_t para tamaños de memoria.
#include <limits>           // Permite consultar límites de tipos numéricos.
#include <stdexcept>        // Excepciones para reportar errores.

// ------------------------------------------------------------
// Wrapper C++ para ejecutar Linear de forma asíncrona.
// Recibe los buffers XRT y las dimensiones de la capa.
// Valida los argumentos y lanza el kernel FPGA mediante XRT,
// devolviendo un xrt::run sin esperar su finalización.
// ------------------------------------------------------------
xrt::run linear_fpga_forward_async(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& output_bo,
    int batch_size,
    int in_features,
    int out_features,
    bool use_bias
)
{
    // Verifica que el tamaño del batch y las cantidades
    // de características de entrada y salida sean positivas
    if (batch_size <= 0 ||
        in_features <= 0 ||
        out_features <= 0) {
        throw std::invalid_argument(
            "Las dimensiones deben ser mayores que cero"
        );
    }

    // ------------------------------------------------------------
    // Función lambda que calcula la memoria requerida en bytes
    // a partir de dos dimensiones.
    // Verifica posibles desbordamientos durante el cálculo
    // de la cantidad de elementos y su conversión a bytes.
    // ------------------------------------------------------------
    const auto buffer_bytes = [](int rows, int columns) {

        // Convierte las dimensiones a std::size_t para calcular
        // las capacidades de memoria
        const auto r = static_cast<std::size_t>(rows);
        const auto c = static_cast<std::size_t>(columns);
        const auto maximum =
            std::numeric_limits<std::size_t>::max();

        // Verifica que la multiplicación de las dimensiones
        // no exceda la capacidad de std::size_t
        if (r > maximum / c) {
            throw std::overflow_error(
                "Cantidad de elementos demasiado grande para Linear"
            );
        }

        const std::size_t count = r * c;

        // Verifica que la conversión de elementos a bytes
        // no produzca un desbordamiento
        if (count > maximum / sizeof(float)) {
            throw std::overflow_error(
                "Tamaño de buffer demasiado grande para Linear"
            );
        }

        return count * sizeof(float);
    };

    // ------------------------------------------------------------
    // Calcula la memoria requerida para los buffers:
    // - Entrada: batch_size x in_features
    // - Pesos: out_features x in_features
    // - Bias: un valor por característica de salida
    // - Salida: batch_size x out_features
    // ------------------------------------------------------------
    const std::size_t input_bytes =
        buffer_bytes(batch_size, in_features);

    const std::size_t weight_bytes =
        buffer_bytes(out_features, in_features);

    const std::size_t bias_bytes =
        buffer_bytes(1, out_features);

    const std::size_t output_bytes =
        buffer_bytes(batch_size, out_features);

    
    // ------------------------------------------------------------
    // Verifica que los buffers XRT tengan capacidad suficiente
    // para almacenar las entradas, pesos, bias y resultados.
    // El buffer de bias es obligatorio incluso cuando
    // use_bias=false, debido a la firma del kernel HLS.
    // ------------------------------------------------------------
    if (input_bo.size() < input_bytes ||
        weight_bo.size() < weight_bytes ||
        bias_bo.size() < bias_bytes ||
        output_bo.size() < output_bytes) {
        throw std::invalid_argument(
            "Los buffers XRT no tienen capacidad suficiente para Linear"
        );
    }

    // Obtiene la instancia compartida de FpgaRuntime.
    auto& runtime = FpgaRuntime::instance();

    // Recupera el kernel linear_forward mediante XRT.
    auto& kernel = runtime.get_kernel("linear_forward");

    // ------------------------------------------------------------
    // Envía los buffers y parámetros al kernel FPGA.
    // Inicia la ejecución mediante XRT y devuelve un xrt::run,
    // permitiendo que el llamador gestione la sincronización.
    // ------------------------------------------------------------
    return kernel(
        input_bo,
        weight_bo,
        bias_bo,
        output_bo,
        batch_size,
        in_features,
        out_features,
        use_bias
    );
}

// ------------------------------------------------------------
// Wrapper C++ para ejecutar Linear de forma síncrona.
// Invoca la función asíncrona y espera la finalización
// del kernel antes de retornar.
// ------------------------------------------------------------
void linear_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& output_bo,
    int batch_size,
    int in_features,
    int out_features,
    bool use_bias
)
{
    
    // Lanza el kernel Linear y obtiene el objeto xrt::run
    // asociado a su ejecución
    auto run = linear_fpga_forward_async(
        input_bo,
        weight_bo,
        bias_bo,
        output_bo,
        batch_size,
        in_features,
        out_features,
        use_bias
    );

    // Bloquea el hilo que invoca la función hasta que finalice
    // el kernel FPGA, sin descargar los resultados a la CPU
    run.wait();
}