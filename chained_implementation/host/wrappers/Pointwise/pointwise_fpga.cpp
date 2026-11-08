#include "pointwise_fpga.hpp" // Declaración de las funciones del wrapper
#include "fpga_runtime.hpp"   // Acceso al runtime y kernels FPGA

#include <cstddef>           // Define std::size_t para tamaños de memoria
#include <initializer_list>  // Permite utilizar listas de dimensiones
#include <limits>            // Permite consultar límites de tipos numéricos
#include <stdexcept>         // Excepciones para reportar errores


// ------------------------------------------------------------
// Wrapper C++ para ejecutar PointwiseConv2D de forma asíncrona
// Recibe los buffers XRT y los parámetros de la convolución 1x1
// Valida los argumentos y lanza el kernel FPGA mediante XRT,
// devolviendo un xrt::run sin esperar su finalización
// ------------------------------------------------------------
xrt::run pointwise_fpga_forward_async(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& output_bo,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    bool use_bias
)
{
    // Verifica que el tamaño del batch, las dimensiones espaciales
    // y las cantidades de canales de entrada y salida sean positivas
    if (batch_size <= 0 ||
        input_channels <= 0 ||
        input_height <= 0 ||
        input_width <= 0 ||
        output_channels <= 0) {
        throw std::invalid_argument(
            "Las dimensiones deben ser mayores que cero"
        );
    }

    /// ------------------------------------------------------------
    // Función lambda que calcula la memoria requerida por un
    // tensor a partir de sus dimensiones
    // Verifica posibles desbordamientos y devuelve el tamaño
    // necesario en bytes para elementos de tipo float
    // ------------------------------------------------------------
    const auto buffer_bytes =
        [](std::initializer_list<int> dimensions) {
            std::size_t count = 1;
            const auto maximum =
                std::numeric_limits<std::size_t>::max();

            // Multiplica las dimensiones del tensor para calcular
            // la cantidad total de elementos, evitando desbordamientos
            for (const int dimension : dimensions) {
               
                const auto value =
                    static_cast<std::size_t>(dimension);

                if (count > maximum / value) {
                    throw std::overflow_error(
                        "Cantidad de elementos demasiado grande "
                        "para PointwiseConv2D"
                    );
                }

                count *= value;
            }

            // Verifica que el cálculo del tamaño en bytes
            // no exceda la capacidad de std::size_t
            if (count > maximum / sizeof(float)) {
                throw std::overflow_error(
                    "Tamano de buffer demasiado grande "
                    "para PointwiseConv2D"
                );
            }

            return count * sizeof(float);
        };

    // Calcula la memoria requerida para el tensor de entrada
    // almacenado en formato NCHW
    const std::size_t input_bytes = buffer_bytes({
        batch_size,
        input_channels,
        input_height,
        input_width
    });

    // Calcula la memoria requerida para los pesos de la
    // convolución 1x1: un peso por cada par de canales
    // de entrada y salida
    const std::size_t weight_bytes = buffer_bytes({
        output_channels,
        input_channels
    });

    // Calcula la memoria requerida para el bias,
    // considerando un valor por canal de salida
    const std::size_t bias_bytes = buffer_bytes({
        output_channels
    });

    // Calcula la memoria requerida para el tensor de salida.
    // PointwiseConv2D conserva la altura y anchura de entrada,
    // pero puede modificar la cantidad de canales.
    const std::size_t output_bytes = buffer_bytes({
        batch_size,
        output_channels,
        input_height,
        input_width
    });

    // ------------------------------------------------------------
    // Verifica que los buffers XRT tengan capacidad suficiente
    // para almacenar las entradas, pesos, bias y resultados
    // El buffer de bias es obligatorio incluso cuando
    // use_bias=false, debido a la firma del kernel HLS
    // ------------------------------------------------------------
    if (input_bo.size() < input_bytes ||
        weight_bo.size() < weight_bytes ||
        bias_bo.size() < bias_bytes ||
        output_bo.size() < output_bytes) {
        throw std::invalid_argument(
            "Buffers XRT insuficientes para PointwiseConv2D"
        );
    }

    // Obtiene la instancia compartida de FpgaRuntime.
    auto& runtime = FpgaRuntime::instance();

    // Obtiene la instancia compartida de FpgaRuntime.
    auto& kernel = runtime.get_kernel(
        "pointwise_conv2d_forward"
    );

    // ------------------------------------------------------------
    // Envía los buffers y parámetros al kernel FPGA.
    // Inicia la ejecución mediante XRT y devuelve un xrt::run,
    // permitiendo que el llamador gestione la sincronización
    // ------------------------------------------------------------
    return kernel(
        input_bo,
        weight_bo,
        bias_bo,
        output_bo,
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_channels,
        use_bias
    );
}

// ------------------------------------------------------------
// Wrapper C++ para ejecutar PointwiseConv2D de forma síncrona.
// Invoca la función asíncrona y espera la finalización
// del kernel antes de retornar.
// ------------------------------------------------------------
void pointwise_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& output_bo,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    bool use_bias
)
{
    // Lanza el kernel PointwiseConv2D y obtiene el objeto xrt::run
    // asociado a su ejecución
    auto run = pointwise_fpga_forward_async(
        input_bo,
        weight_bo,
        bias_bo,
        output_bo,
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_channels,
        use_bias
    );

    // Bloquea el hilo que invoca la función hasta que finalice
    // el kernel FPGA, sin descargar los resultados a la CPU
    run.wait();
}