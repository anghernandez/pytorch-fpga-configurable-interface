#include "linear_fpga.hpp"
#include "fpga_runtime.hpp"

#include <cstddef>
#include <stdexcept>

#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>


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
    // ============================================================
    // 1. VALIDACIÓN DE DIMENSIONES
    // ============================================================

    if (
        batch_size <= 0 ||
        in_features <= 0 ||
        out_features <= 0
    ) {
        throw std::invalid_argument(
            "Las dimensiones deben ser mayores que cero"
        );
    }

    // Cantidad de elementos de cada argumento del kernel Linear.

    const std::size_t input_size =
        static_cast<std::size_t>(batch_size)
        * static_cast<std::size_t>(in_features);

    const std::size_t weight_size =
        static_cast<std::size_t>(out_features)
        * static_cast<std::size_t>(in_features);

    const std::size_t bias_size =
        static_cast<std::size_t>(out_features);

    const std::size_t output_size =
        static_cast<std::size_t>(batch_size)
        * static_cast<std::size_t>(out_features);


    // ============================================================
    // 2. VALIDACIÓN DE CAPACIDAD DE BUFFERS EXTERNOS
    // ============================================================

    // Los buffers son administrados externamente.
    // Se verifica que tengan suficiente capacidad para los datos.

    // bias_bo sigue siendo necesario aunque use_bias=false,
    // porque forma parte de los argumentos del kernel HLS.

    if (
        input_bo.size() < input_size * sizeof(float) ||
        weight_bo.size() < weight_size * sizeof(float) ||
        bias_bo.size() < bias_size * sizeof(float) ||
        output_bo.size() < output_size * sizeof(float)
    ) {
        throw std::invalid_argument(
            "Los buffers XRT no tienen capacidad suficiente para Linear"
        );
    }


    // ============================================================
    // 3. OBTENCIÓN DEL RUNTIME Y DEL KERNEL XRT
    // ============================================================

    // Obtiene la instancia compartida del runtime.

    auto& runtime = FpgaRuntime::instance();

    // Obtiene el dispositivo administrado por el runtime.

    xrt::device& device = runtime.device();

    // Identifica el kernel Linear dentro del XCLBIN cargado.

    xrt::kernel kernel(
        device,
        runtime.uuid(),
        "linear_forward"
    );


    // ============================================================
    // 4. UTILIZACIÓN DE BUFFERS EXTERNOS
    // ============================================================

    // Los buffers recibidos ya existen.
    //
    // input_bo:  datos de entrada.
    // weight_bo: pesos de la capa.
    // bias_bo:   bias de la capa.
    // output_bo: resultados de la operación.
    //
    // No se crean buffers locales.
    // No se realizan transferencias CPU -> FPGA.
    // No se realizan transferencias FPGA -> CPU.
    //
    // Los pesos y bias pueden permanecer disponibles entre
    // ejecuciones sin necesidad de cargarlos nuevamente.
    //
    // La salida puede utilizarse directamente como entrada
    // de otra operación.


    // ============================================================
    // 5. EJECUCIÓN DEL KERNEL HLS
    // ============================================================

    // Se proporcionan las referencias a los buffers y
    // los parámetros escalares necesarios para Linear.

    auto run = kernel(
        input_bo,
        weight_bo,
        bias_bo,
        output_bo,
        batch_size,
        in_features,
        out_features,
        use_bias
    );


    // ============================================================
    // 6. FINALIZACIÓN DE LA EJECUCIÓN
    // ============================================================

    // Espera a que el kernel termine.
    // No transfiere los resultados a la CPU.

    run.wait();

    // output_bo conserva el resultado y puede utilizarse
    // posteriormente por otro kernel.
}