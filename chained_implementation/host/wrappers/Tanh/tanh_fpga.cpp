#include "tanh_fpga.hpp"    // Declaración de las funciones del wrapper
#include "fpga_runtime.hpp" // Acceso al runtime y kernels FPGA

#include <cstddef>          // Define std::size_t para tamaños de memoria
#include <limits>           // Permite consultar límites de tipos numéricos
#include <stdexcept>        // Excepciones para reportar errores


// ------------------------------------------------------------
// Wrapper C++ para ejecutar Tanh de forma asíncrona.
// Recibe los buffers XRT de entrada y salida, junto con
// la cantidad de elementos que deben procesarse.
// Valida los argumentos y lanza el kernel FPGA mediante XRT,
// devolviendo un xrt::run sin esperar su finalización.
// ------------------------------------------------------------
xrt::run tanh_fpga_forward_async(
    xrt::bo& input_bo,
    xrt::bo& output_bo,
    int size
)
{   
    // Verifica que la cantidad de elementos sea mayor que cero.
    if (size <= 0) {
        throw std::invalid_argument(
            "size debe ser mayor que cero"
        );
    }

    // Convierte la cantidad de elementos a std::size_t
    // para realizar los cálculos de capacidad de memoria.
    const auto count = static_cast<std::size_t>(size);

    // Verifica que el cálculo del tamaño en bytes
    // no exceda la capacidad de std::size_t
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error(
            "El tamaño solicitado excede la capacidad representable"
        );
    }

    // Verifica que el cálculo del tamaño en bytes
    // no exceda la capacidad de std::size_t
    const std::size_t bytes = count * sizeof(float);

    // ------------------------------------------------------------
    // Verifica que los buffers XRT de entrada y salida
    // tengan capacidad suficiente para almacenar los elementos
    // No crea ni redimensiona buffers
    // ------------------------------------------------------------
    if (input_bo.size() < bytes || output_bo.size() < bytes) {
        throw std::invalid_argument(
            "Los buffers XRT no tienen capacidad suficiente"
        );
    }

    // Obtiene la instancia compartida de FpgaRuntime.
    auto& runtime = FpgaRuntime::instance();

    // Recupera el kernel tanh_forward mediante XRT.
    auto& kernel = runtime.get_kernel("tanh_forward");

    // ------------------------------------------------------------
    // Envía los buffers y el tamaño al kernel FPGA.
    // Inicia la ejecución mediante XRT y devuelve un xrt::run,
    // permitiendo que el llamador gestione la sincronización
    // ------------------------------------------------------------
    return kernel(input_bo, output_bo, size);
}

// Lanza el kernel Tanh y obtiene el objeto xrt::run
// asociado a su ejecución
void tanh_fpga_forward(
    xrt::bo& input_bo,
    xrt::bo& output_bo,
    int size
)
{
    // Lanza el kernel Tanh y obtiene el objeto xrt::run
    // asociado a su ejecución
    auto run = tanh_fpga_forward_async(
        input_bo,
        output_bo,
        size
    );

    // Bloquea el hilo que invoca la función hasta que finalice
    // el kernel FPGA, sin descargar los resultados a la CPU
    run.wait();
}