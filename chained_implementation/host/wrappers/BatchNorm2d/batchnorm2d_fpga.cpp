#include "batchnorm2d_fpga.hpp" // Declaración de las funciones del wrapper
#include "fpga_runtime.hpp"    // Acceso al runtime y kernels FPGA

#include <cmath>              // Funciones matemáticas y validación de epsilon
#include <cstddef>            // Define std::size_t para tamaños de memoria
#include <initializer_list>   // Permite recibir listas de dimensiones
#include <limits>             // Permite consultar límites de tipos numéricos
#include <stdexcept>          // Excepciones para reportar argumentos inválidos

// ------------------------------------------------------------
// Wrapper C++ para ejecutar BatchNorm2D de forma asíncrona
// Recibe los buffers XRT de entrada, parámetros y salida
// Valida las dimensiones y capacidades antes de lanzar
// el kernel FPGA, sin esperar su finalización
// ------------------------------------------------------------
xrt::run batchnorm2d_fpga_forward_async(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& running_mean_bo,
    xrt::bo& running_var_bo,
    xrt::bo& output_bo,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    float eps,
    bool affine,
    bool use_bias
)
{

    // Verifica que las dimensiones del tensor de entrada
    // sean mayores que cero
    if (batch_size <= 0 ||
        channels <= 0 ||
        input_height <= 0 ||
        input_width <= 0) {
        throw std::invalid_argument(
            "Las dimensiones deben ser mayores que cero"
        );
    }

    // Verifica que epsilon sea positivo y finito
    // Evita utilizar valores inválidos en la normalización
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument(
            "eps debe ser positivo y finito"
        );
    }

    // ------------------------------------------------------------
    // Función lambda para calcular la memoria requerida por
    // un tensor a partir de sus dimensiones.
    // Verifica posibles desbordamientos y devuelve el tamaño
    // necesario en bytes para elementos float
    // ------------------------------------------------------------
    const auto buffer_bytes =
        [](std::initializer_list<int> dimensions) {
            std::size_t count = 1;

            const auto maximum =
                std::numeric_limits<std::size_t>::max();

            // Multiplica las dimensiones para obtener la cantidad total
            // de elementos, verificando que no exista desbordamiento
            for (const int dimension : dimensions) {
              
                const auto value =
                    static_cast<std::size_t>(dimension);
            
                if (count > maximum / value) {
                    throw std::overflow_error(
                        "Cantidad de elementos demasiado grande "
                        "para BatchNorm2D"
                    );
                }

                count *= value;
            }

            // Verifica que la conversión de elementos a bytes
            // no exceda la capacidad de std::size_t
            if (count > maximum / sizeof(float)) {
                throw std::overflow_error(
                    "Tamano de buffer demasiado grande "
                    "para BatchNorm2D"
                );
            }

            return count * sizeof(float);
        };

    
    // ------------------------------------------------------------
    // Calcula la memoria requerida para los tensores de entrada
    // y salida en formato NCHW.
    // BatchNorm2D conserva las dimensiones del tensor.
    // ------------------------------------------------------------
    const std::size_t data_bytes = buffer_bytes({
        batch_size,
        channels,
        input_height,
        input_width
    });

    // Calcula la memoria requerida para weight, bias,
    // running_mean y running_var, con un valor por canal
    const std::size_t param_bytes = buffer_bytes({
        channels
    });

   
    // ------------------------------------------------------------
    // Verifica que todos los buffers XRT tengan capacidad suficiente.
    // Los buffers de parámetros deben existir incluso cuando
    // affine o use_bias están deshabilitados, debido a la firma HLS.
    // ------------------------------------------------------------
    if (input_bo.size() < data_bytes ||
        output_bo.size() < data_bytes ||
        weight_bo.size() < param_bytes ||
        bias_bo.size() < param_bytes ||
        running_mean_bo.size() < param_bytes ||
        running_var_bo.size() < param_bytes) {
        throw std::invalid_argument(
            "Buffers XRT insuficientes para BatchNorm2D"
        );
    }

    // Obtiene la instancia compartida de FpgaRuntime.
    auto& runtime = FpgaRuntime::instance();

    // Recupera el kernel batchnorm2d_forward mediante XRT.
    auto& kernel = runtime.get_kernel(
        "batchnorm2d_forward"
    );

    // ------------------------------------------------------------
    // Envía los buffers y parámetros al kernel FPGA.
    // Inicia la ejecución y devuelve un objeto xrt::run,
    // permitiendo que el llamador controle la sincronización.
    // ------------------------------------------------------------
    return kernel(
        input_bo,
        weight_bo,
        bias_bo,
        running_mean_bo,
        running_var_bo,
        output_bo,
        batch_size,
        channels,
        input_height,
        input_width,
        eps,
        affine,
        use_bias
    );
}



// ------------------------------------------------------------
// Wrapper C++ para ejecutar BatchNorm2D de forma síncrona.
// Invoca la función asíncrona y espera la finalización
// del kernel antes de retornar.
// ------------------------------------------------------------
void batchnorm2d_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& weight_bo,
    xrt::bo& bias_bo,
    xrt::bo& running_mean_bo,
    xrt::bo& running_var_bo,
    xrt::bo& output_bo,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    float eps,
    bool affine,
    bool use_bias
)
{
    // Lanza el kernel FPGA y obtiene el objeto xrt::run
    // asociado a su ejecución.
    auto run = batchnorm2d_fpga_forward_async(
        input_bo,
        weight_bo,
        bias_bo,
        running_mean_bo,
        running_var_bo,
        output_bo,
        batch_size,
        channels,
        input_height,
        input_width,
        eps,
        affine,
        use_bias
    );

    // Bloquea el hilo que invoca la función hasta que finalice
    // el kernel FPGA, sin transferir los resultados a la CPU.
    run.wait();
}