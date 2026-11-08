#include "conv2d_fpga.hpp"  // Declaración de las funciones del wrapper.
#include "fpga_runtime.hpp" // Acceso al runtime y a los kernels FPGA.

#include <cstddef>          // Define std::size_t para tamaños de memoria.
#include <cstdint>          // Define tipos enteros como std::int64_t.
#include <initializer_list> // Permite utilizar listas de dimensiones.
#include <limits>           // Permite consultar límites de tipos numéricos.
#include <stdexcept>        // Excepciones para reportar errores.


// ------------------------------------------------------------
// Wrapper C++ para ejecutar Conv2D de forma asíncrona
// Recibe los buffers XRT y los parámetros de convolución.
// Valida los argumentos y lanza el kernel FPGA mediante XRT,
// devolviendo un xrt::run sin esperar su finalización
// ------------------------------------------------------------
xrt::run conv2d_fpga_forward_async(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& output_bo,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool use_bias
)
{
    // ------------------------------------------------------------
    // Verifica que las dimensiones de entrada, canales, kernel
    // y stride sean positivos, y que el padding no sea negativo.
    // ------------------------------------------------------------
    if (batch_size <= 0 ||
        input_channels <= 0 ||
        input_height <= 0 ||
        input_width <= 0 ||
        output_channels <= 0 ||
        kernel_height <= 0 ||
        kernel_width <= 0 ||
        stride_height <= 0 ||
        stride_width <= 0 ||
        padding_height < 0 ||
        padding_width < 0) {
        throw std::invalid_argument(
            "Dimensiones, kernel, stride o padding invalidos"
        );
    }

    // ------------------------------------------------------------
    // Calcula los numeradores de las dimensiones de salida,
    // considerando el tamaño de entrada, kernel y padding.
    // Utiliza enteros de 64 bits para evitar desbordamientos
    // en los cálculos intermedios.
    // ------------------------------------------------------------
    const std::int64_t numerator_h =
        static_cast<std::int64_t>(input_height) +
        2LL * padding_height - kernel_height;

    const std::int64_t numerator_w =
        static_cast<std::int64_t>(input_width) +
        2LL * padding_width - kernel_width;

    // Verifica que la configuración permita obtener
    // dimensiones espaciales de salida válidas
    if (numerator_h < 0 || numerator_w < 0) {
        throw std::invalid_argument(
            "La convolucion produce dimensiones invalidas"
        );
    }

    const std::int64_t expected_h =
        numerator_h / stride_height + 1;

    const std::int64_t expected_w =
        numerator_w / stride_width + 1;

    // Comprueba que las dimensiones de salida proporcionadas
    // coincidan con las calculadas para la convolución
    if (output_height <= 0 ||
        output_width <= 0 ||
        static_cast<std::int64_t>(output_height) != expected_h ||
        static_cast<std::int64_t>(output_width) != expected_w) {
        throw std::invalid_argument(
            "Las dimensiones de salida son incorrectas"
        );
    }

    // ------------------------------------------------------------
    // Función lambda que calcula la memoria requerida por un
    // tensor a partir de sus dimensiones
    // Comprueba posibles desbordamientos y devuelve el tamaño
    // necesario en bytes para datos de tipo float
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
                        "Cantidad de elementos demasiado grande para Conv2D"
                    );
                }

                count *= value;
            }

            // Verifica que la conversión de elementos a bytes
            // no exceda la capacidad de std::size_t
            if (count > maximum / sizeof(float)) {
                throw std::overflow_error(
                    "Tamano de buffer demasiado grande para Conv2D"
                );
            }

            return count * sizeof(float);
        };

    // Calcula la memoria requerida para los pesos del filtro
    // en formato OIHW: canales de salida, canales de entrada,
    // altura y anchura del kernel
    const std::size_t input_bytes = buffer_bytes({
        batch_size,
        input_channels,
        input_height,
        input_width
    });

    // Pesos: canales de salida, canales de entrada, Kh y Kw.
    const std::size_t weight_bytes = buffer_bytes({
        output_channels,
        input_channels,
        kernel_height,
        kernel_width
    });

    // Calcula la memoria requerida para el bias,
    // considerando un valor por canal de salida
    const std::size_t bias_bytes = buffer_bytes({
        output_channels
    });

    // Calcula la memoria requerida para las activaciones
    // de salida almacenadas en formato NCHW
    const std::size_t output_bytes = buffer_bytes({
        batch_size,
        output_channels,
        output_height,
        output_width
    });

    // ------------------------------------------------------------
    // Verifica que los buffers XRT tengan capacidad suficiente
    // para almacenar los datos de entrada, pesos, bias y salida.
    // El buffer de bias es obligatorio incluso si use_bias=false,
    // debido a la firma del kernel HLS.
    // ------------------------------------------------------------
    if (input_bo.size() < input_bytes ||
        weight_bo.size() < weight_bytes ||
        bias_bo.size() < bias_bytes ||
        output_bo.size() < output_bytes) {
        throw std::invalid_argument(
            "Buffers XRT insuficientes para Conv2D"
        );
    }

    // Obtiene la instancia compartida de FpgaRuntime.
    auto& runtime = FpgaRuntime::instance();

    // Recupera el kernel conv2d_forward mediante XRT.
    auto& kernel = runtime.get_kernel("conv2d_forward");

    
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
}

// ------------------------------------------------------------
// Wrapper C++ para ejecutar Conv2D de forma síncrona.
// Invoca la función asíncrona y espera la finalización
// del kernel antes de retornar.
// ------------------------------------------------------------
void conv2d_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& output_bo,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool use_bias
)
{
    // Lanza el kernel Conv2D y obtiene el objeto xrt::run
    // que representa su ejecución
    auto run = conv2d_fpga_forward_async(
        input_bo,
        weight_bo,
        bias_bo,
        output_bo,
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

    // Bloquea el hilo que invoca la función hasta que finalice
    // la ejecución del kernel FPGA, sin descargar los resultados
    run.wait();
}