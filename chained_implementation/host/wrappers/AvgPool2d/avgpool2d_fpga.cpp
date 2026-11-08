#include "avgpool2d_fpga.hpp"  // Declaración de las funciones del wrapper.
#include "fpga_runtime.hpp"   // Acceso al runtime y a los kernels FPGA.

#include <cstddef>           // Define std::size_t para tamaños de memoria.
#include <cstdint>           // Define tipos enteros como std::int64_t.
#include <initializer_list>  // Permite recibir listas de dimensiones.
#include <limits>            // Permite consultar los límites de tipos numéricos.
#include <stdexcept>         // Proporciona excepciones para validar errores.

// ------------------------------------------------------------
// Wrapper C++ para ejecutar AvgPool2D de forma asíncrona.
// Recibe buffers XRT y parámetros de configuración,
// valida los argumentos y lanza el kernel en la FPGA.
// Devuelve un xrt::run sin esperar su finalización.
// ------------------------------------------------------------
xrt::run avgpool2d_fpga_forward_async(
    xrt::bo& input_bo,
    xrt::bo& output_bo,
    int batch_size,
    int input_channels,
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
    bool count_include_pad,
    bool use_divisor_override,
    int divisor_override
)
{
   
    // ------------------------------------------------------------
    // Verifica que las dimensiones del tensor, tamaño del kernel
    // y stride sean positivos antes de ejecutar la operación.
    // ------------------------------------------------------------
    if (batch_size <= 0 ||
        input_channels <= 0 ||
        input_height <= 0 ||
        input_width <= 0 ||
        output_height <= 0 ||
        output_width <= 0 ||
        kernel_height <= 0 ||
        kernel_width <= 0 ||
        stride_height <= 0 ||
        stride_width <= 0) {
        throw std::invalid_argument(
            "Dimensiones, kernel y stride deben ser positivos"
        );
    }

    // Verifica que los valores de padding no sean negativos.
    if (padding_height < 0 || padding_width < 0) {
        throw std::invalid_argument(
            "El padding no puede ser negativo"
        );
    }

    // Comprueba que el divisor personalizado sea positivo
    // cuando su utilización está habilitada.
    if (use_divisor_override && divisor_override <= 0) {
        throw std::invalid_argument(
            "divisor_override debe ser positivo cuando se utiliza"
        );
    }

   
    // ------------------------------------------------------------
    // Calcula los numeradores de la fórmula de dimensiones
    // de salida de AvgPool2D, considerando kernel, padding
    // y dimensiones de entrada.
    // Utiliza enteros de 64 bits para evitar desbordamientos
    // en las operaciones intermedias.
    // ------------------------------------------------------------
    const std::int64_t numerator_h =
        static_cast<std::int64_t>(input_height) +
        2LL * padding_height - kernel_height;

    const std::int64_t numerator_w =
        static_cast<std::int64_t>(input_width) +
        2LL * padding_width - kernel_width;

    // Verifica que las dimensiones de salida coincidan con
    // las calculadas mediante división entera (sin ceil_mode)
    if (numerator_h < 0 ||
        numerator_w < 0 ||
        numerator_h / stride_height + 1 != output_height ||
        numerator_w / stride_width + 1 != output_width) {
        throw std::invalid_argument(
            "Las dimensiones de salida no coinciden con AvgPool2D"
        );
    }

    // ------------------------------------------------------------
    // Función lambda para calcular la memoria requerida por un
    // tensor a partir de sus dimensiones.
    // Comprueba que las multiplicaciones no excedan la capacidad
    // de std::size_t y devuelve el tamaño necesario en bytes.
    // ------------------------------------------------------------
    const auto buffer_bytes =
        [](std::initializer_list<int> dimensions) {
            std::size_t count = 1;

            const auto maximum =
                std::numeric_limits<std::size_t>::max();

            // Recorre las dimensiones del tensor y calcula la cantidad
            // total de elementos, verificando posibles desbordamientos
            for (const int dimension : dimensions) {
                const auto value =
                    static_cast<std::size_t>(dimension);
                
                // Verifica que la conversión de elementos a bytes no
                // produzca un desbordamiento
                if (count > maximum / value) {
                    throw std::overflow_error(
                        "Cantidad de elementos demasiado grande para AvgPool2D"
                    );
                }

                count *= value;
            }

            if (count > maximum / sizeof(float)) {
                throw std::overflow_error(
                    "Tamano de buffer demasiado grande para AvgPool2D"
                );
            }

            // Devuelve la capacidad requerida en bytes para datos float.
            return count * sizeof(float);
        };

    
    // ------------------------------------------------------------
    // Calcula la memoria requerida para los tensores de entrada
    // y salida utilizando el formato NCHW
    // AvgPool2D conserva la cantidad de canales
    // ------------------------------------------------------------
    const std::size_t input_bytes = buffer_bytes({
        batch_size,
        input_channels,
        input_height,
        input_width
    });

    const std::size_t output_bytes = buffer_bytes({
        batch_size,
        input_channels,
        output_height,
        output_width
    });

    // Verifica que los Buffer Objects (BO) existentes tengan
    // capacidad suficiente para almacenar los tensores
    // No crea ni redimensiona buffers
    if (input_bo.size() < input_bytes ||
        output_bo.size() < output_bytes) {
        throw std::invalid_argument(
            "Los buffers XRT no tienen capacidad suficiente para AvgPool2D"
        );
    }

    // ------------------------------------------------------------
    // Obtiene la instancia compartida de FpgaRuntime y recupera
    // el kernel AvgPool2D previamente registrado en XRT.
    // ------------------------------------------------------------  
    auto& runtime = FpgaRuntime::instance();
    auto& kernel = runtime.get_kernel("avg_pool2d_forward");

    // ------------------------------------------------------------
    // Envía los buffers y parámetros al kernel FPGA
    // Inicia la ejecución mediante XRT y devuelve un xrt::run,
    // permitiendo que el llamador gestione la sincronización
    // ------------------------------------------------------------
    return kernel(
        input_bo,
        output_bo,
        batch_size,
        input_channels,
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
        count_include_pad,
        use_divisor_override,
        divisor_override
    );
}


// ------------------------------------------------------------
// Wrapper C++ para ejecutar AvgPool2D de forma síncrona
// Utiliza la función asíncrona para iniciar la operación
// y espera su finalización antes de retornar
// ------------------------------------------------------------
void avgpool2d_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& output_bo,
    int batch_size,
    int input_channels,
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
    bool count_include_pad,
    bool use_divisor_override,
    int divisor_override
)
{
    // Lanza el kernel y obtiene el objeto xrt::run
    // que representa su ejecución
    auto run = avgpool2d_fpga_forward_async(
        input_bo,
        output_bo,
        batch_size,
        input_channels,
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
        count_include_pad,
        use_divisor_override,
        divisor_override
    );

    // Espera hasta que finalice la ejecución del kernel FPGA
    // No realiza una transferencia de resultados hacia la CPU
    run.wait(); // Bloquea el hilo hasta que termine el kernel
}