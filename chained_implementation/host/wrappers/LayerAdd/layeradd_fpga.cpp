#include "layeradd_fpga.hpp" // Declaración de las funciones del wrapper.
#include "fpga_runtime.hpp"  // Acceso al runtime y kernels FPGA.

#include <cmath>             // Permite verificar si alpha es finito.
#include <cstddef>           // Define std::size_t para tamaños de memoria.
#include <limits>            // Permite consultar límites de tipos numéricos.
#include <stdexcept>         // Excepciones para reportar errores.


// ------------------------------------------------------------
// Wrapper C++ para ejecutar LayerAdd de forma asíncrona.
// Recibe dos buffers de entrada, uno de salida y los
// parámetros size y alpha
// Valida los argumentos y lanza el kernel FPGA mediante XRT,
// devolviendo un xrt::run sin esperar su finalización
// ------------------------------------------------------------
xrt::run layeradd_fpga_forward_async(
    xrt::bo& input1_bo,
    xrt::bo& input2_bo,
    xrt::bo& output_bo,
    int size,
    float alpha
)
{
    // Verifica que la cantidad de elementos sea mayor que cero.
    if (size <= 0) {
        throw std::invalid_argument(
            "size debe ser mayor que cero"
        );
    }

    // Verifica que alpha sea un valor numérico finito,
    // rechazando valores infinitos o NaN
    if (!std::isfinite(alpha)) {
        throw std::invalid_argument(
            "alpha debe ser finito"
        );
    }

    // Convierte la cantidad de elementos a std::size_t
    // para realizar cálculos de capacidad de memoria.
    const auto count = static_cast<std::size_t>(size);

    // Verifica que el cálculo del tamaño en bytes
    // no exceda la capacidad de std::size_t
    if (count >
        std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error(
            "Tamano de buffer demasiado grande para LayerAdd"
        );
    }

    // Calcula la memoria requerida en bytes para cada tensor,
    // considerando que todos contienen elementos float.
    const std::size_t bytes = count * sizeof(float);

    // ------------------------------------------------------------
    // Verifica que los dos buffers de entrada y el de salida
    // tengan capacidad suficiente para almacenar los elementos
    // No crea ni redimensiona buffers.
    // ------------------------------------------------------------
    if (input1_bo.size() < bytes ||
        input2_bo.size() < bytes ||
        output_bo.size() < bytes) {
        throw std::invalid_argument(
            "Buffers XRT insuficientes para LayerAdd"
        );
    }

    // Obtiene la instancia compartida de FpgaRuntime.
    auto& runtime = FpgaRuntime::instance();

    // Recupera el kernel layer_add_forward mediante XRT.
    auto& kernel = runtime.get_kernel(
        "layer_add_forward"
    );

    // ------------------------------------------------------------
    // Envía los buffers y parámetros al kernel FPGA.
    // Inicia la ejecución mediante XRT y devuelve un xrt::run,
    // permitiendo que el llamador gestione la sincronización.
    // ------------------------------------------------------------
    return kernel(
        input1_bo,
        input2_bo,
        output_bo,
        size,
        alpha
    );
}


// ------------------------------------------------------------
// Wrapper C++ para ejecutar LayerAdd de forma síncrona.
// Invoca la función asíncrona y espera la finalización
// del kernel antes de retornar.
// ------------------------------------------------------------
void layeradd_fpga_forward(
    xrt::bo& input1_bo,
    xrt::bo& input2_bo,
    xrt::bo& output_bo,
    int size,
    float alpha
)
{
    // Lanza el kernel LayerAdd y obtiene el objeto xrt::run
    // asociado a su ejecución
    auto run = layeradd_fpga_forward_async(
        input1_bo,
        input2_bo,
        output_bo,
        size,
        alpha
    );

    // Bloquea el hilo que invoca la función hasta que finalice
    // el kernel FPGA, sin descargar los resultados a la CPU
    run.wait();
}