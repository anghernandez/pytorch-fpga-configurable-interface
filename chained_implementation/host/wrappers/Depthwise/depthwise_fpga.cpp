#include "depthwise_fpga.hpp" // Declaración de las funciones del wrapper.
#include "fpga_runtime.hpp"  // Acceso al runtime y kernels FPGA.

#include <cstddef>          // Define std::size_t para tamaños de memoria.
#include <cstdint>          // Define std::int64_t para enteros de 64 bits.
#include <initializer_list> // Permite utilizar listas de dimensiones.
#include <limits>           // Permite consultar límites de tipos numéricos.
#include <stdexcept>        // Excepciones para reportar errores.

// ------------------------------------------------------------
// Wrapper C++ para ejecutar DepthwiseConv2D de forma asíncrona.
// Recibe los buffers XRT y los parámetros de convolución
// Valida los argumentos y lanza el kernel FPGA mediante XRT,
// devolviendo un xrt::run sin esperar su finalización
// ------------------------------------------------------------
xrt::run depthwise_fpga_forward_async(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& output_bo,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
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
    // Verifica que las dimensiones del tensor, kernel y stride
    // sean positivas, y que el padding no sea negativo.
    // ------------------------------------------------------------
    if (batch_size <= 0 ||
        channels <= 0 ||
        input_height <= 0 ||
        input_width <= 0 ||
        output_height <= 0 ||
        output_width <= 0 ||
        kernel_height <= 0 ||
        kernel_width <= 0 ||
        stride_height <= 0 ||
        stride_width <= 0 ||
        padding_height < 0 ||
        padding_width < 0) {
        throw std::invalid_argument(
            "Dimensiones, stride o padding invalidos "
            "en DepthwiseConv2D"
        );
    }

    // ------------------------------------------------------------
    // Calcula los numeradores de las dimensiones de salida
    // considerando el tamaño de entrada, kernel y padding
    // Utiliza enteros de 64 bits para los cálculos intermedios
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
            "DepthwiseConv2D produce dimensiones invalidas"
        );
    }

    // Comprueba que las dimensiones de salida coincidan con
    // las calculadas mediante la fórmula de convolución
    if (numerator_h / stride_height + 1 != output_height ||
        numerator_w / stride_width + 1 != output_width) {
        throw std::invalid_argument(
            "Dimensiones de salida incompatibles "
            "con DepthwiseConv2D"
        );
    }

    // ------------------------------------------------------------
    // Función lambda que calcula la memoria requerida por un
    // tensor a partir de sus dimensiones
    // Verifica posibles desbordamientos y devuelve el tamaño
    // necesario en bytes para elementos de tipo float.
    // ------------------------------------------------------------
    const auto buffer_bytes =
        [](std::initializer_list<int> dimensions) {
            std::size_t count = 1;
            const auto maximum =
                std::numeric_limits<std::size_t>::max();

            // Multiplica las dimensiones para obtener la cantidad total
            // de elementos, verificando posibles desbordamientos
            for (const int dimension : dimensions) {
                
                const auto value =
                    static_cast<std::size_t>(dimension);

                if (count > maximum / value) {
                    throw std::overflow_error(
                        "Cantidad de elementos demasiado grande "
                        "para DepthwiseConv2D"
                    );
                }

                count *= value;
            }

            // Calcula la memoria requerida para el tensor de entrada
            // almacenado en formato NCHW
            if (count > maximum / sizeof(float)) {
                throw std::overflow_error(
                    "Tamano de buffer demasiado grande "
                    "para DepthwiseConv2D"
                );
            }

            return count * sizeof(float);
        };

    // Calcula la memoria requerida para el tensor de entrada
    // almacenado en formato NCHW
    const std::size_t input_bytes = buffer_bytes({
        batch_size, channels, input_height, input_width
    });

    // Calcula la memoria requerida para los pesos
    // DepthwiseConv2D utiliza un filtro espacial por canal
    const std::size_t weight_bytes = buffer_bytes({
        channels, kernel_height, kernel_width
    });

    // Calcula la memoria requerida para el bias,
    // considerando un valor por canal
    const std::size_t bias_bytes = buffer_bytes({
        channels
    });

    // Calcula la memoria requerida para el tensor de salida.
    // DepthwiseConv2D conserva la cantidad de canales
    const std::size_t output_bytes = buffer_bytes({
        batch_size, channels, output_height, output_width
    });

    // ------------------------------------------------------------
    // Verifica que los buffers XRT tengan capacidad suficiente
    // para almacenar entrada, pesos, bias y salida
    // El buffer de bias es obligatorio incluso cuando
    // use_bias=false, debido a la firma del kernel HLS
    // ------------------------------------------------------------
    if (input_bo.size() < input_bytes ||
        weight_bo.size() < weight_bytes ||
        bias_bo.size() < bias_bytes ||
        output_bo.size() < output_bytes) {
        throw std::invalid_argument(
            "Buffers XRT insuficientes para DepthwiseConv2D"
        );
    }

    // Obtiene la instancia compartida de FpgaRuntime
    auto& runtime = FpgaRuntime::instance();

    // Recupera el kernel depthwise_conv2d_forward mediante XRT
    auto& kernel = runtime.get_kernel(
        "depthwise_conv2d_forward"
    );

    // ------------------------------------------------------------
    // Envía los buffers y parámetros al kernel FPGA.
    // Inicia la ejecución mediante XRT y devuelve un xrt::run,
    // permitiendo que el llamador controle la sincronización.
    // ------------------------------------------------------------
    return kernel(
        input_bo,
        weight_bo,
        bias_bo,
        output_bo,
        batch_size,
        channels,
        input_height,
        input_width,
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
// Wrapper C++ para ejecutar DepthwiseConv2D de forma síncrona
// Invoca la función asíncrona y espera la finalización
// del kernel antes de retornar
// ------------------------------------------------------------
void depthwise_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& output_bo,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
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
    // Lanza el kernel FPGA y obtiene el objeto xrt::run
    // asociado a su ejecución
    auto run = depthwise_fpga_forward_async(
        input_bo,
        weight_bo,
        bias_bo,
        output_bo,
        batch_size,
        channels,
        input_height,
        input_width,
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
    // el kernel FPGA, sin descargar los resultados a la CPU
    run.wait();
}