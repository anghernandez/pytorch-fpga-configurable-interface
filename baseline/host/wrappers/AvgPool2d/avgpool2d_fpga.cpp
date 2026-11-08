/* Wrapper/ Host: corre en la CPU de la Kria
 * actua como puente entre el código de alto nivel Pytorch y código C++ de host
 * 
 * Este código prepara los datos, configura la ejecución y solicita a XRT 
 * que ejecute el kernel que está implementado en la FPGA.
 */

#include "avgpool2d_fpga.hpp"

// Permite el runtime común de la FPGA, que maneja la carga del bitstream y la inicialización del dispositivo.
#include "fpga_runtime.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>


/* xrt::device -> representa el dispositivo FPGA. 
 * xrt::kernel -> representa el kernel que queremos ejecutar. 
 * xrt::bo -> representa buffers de memoria administrados por XRT.
 */
#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>

//Medir con instrumentación
#include <chrono>
#include <iostream>
#include "fpga_profile.hpp"

//esta función permite ejecutar AvgPool2d usando la FPGA
void avgpool2d_fpga_forward( 
    const float* input,
    float* output,
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
    //validaciones, antes de comunicarse con la FPGA se verfica que los datos y parametros tengan sentido.
    if (input == nullptr) {
        throw std::invalid_argument(
            "input no puede ser nullptr"
        );
    }

    if (output == nullptr) {
        throw std::invalid_argument(
            "output no puede ser nullptr"
        );
    }

    if (
        batch_size <= 0 ||
        input_channels <= 0 ||
        input_height <= 0 ||
        input_width <= 0
    ) {
        throw std::invalid_argument(
            "Las dimensiones de entrada deben ser mayores que cero"
        );
    }

    if (
        output_height <= 0 ||
        output_width <= 0
    ) {
        throw std::invalid_argument(
            "Las dimensiones de salida deben ser mayores que cero"
        );
    }

    if (
        kernel_height <= 0 ||
        kernel_width <= 0
    ) {
        throw std::invalid_argument(
            "El tamaño del kernel debe ser mayor que cero"
        );
    }

    // El stride determina cuánto se desplaza la ventana de pooling.
    if (
        stride_height <= 0 ||
        stride_width <= 0
    ) {
        throw std::invalid_argument(
            "El stride debe ser mayor que cero"
        );
    }


    // El padding representa elementos agregados alrededor de la entrada.
    // No tiene sentido utilizar un padding negativo
    if (
        padding_height < 0 ||
        padding_width < 0
    ) {
        throw std::invalid_argument(
            "El padding no puede ser negativo"
        );
    }

    // Si se utiliza un divisor debe ser positivo
    if (
        use_divisor_override &&
        divisor_override <= 0
    ) {
        throw std::invalid_argument(
            "divisor_override debe ser mayor que cero"
        );
    }

    // Validaciones de salida, se verifica que output_height y output_width sean consistentes con los parámetros de entrada, kernel, stride y padding.
    const int expected_output_height =
        (
            input_height
            + 2 * padding_height
            - kernel_height
        ) / stride_height
        + 1;

    const int expected_output_width =
        (
            input_width
            + 2 * padding_width
            - kernel_width
        ) / stride_width
        + 1;

    // Comparamos las dimensiones recibidas con las que deberían producirse
    if (
        output_height != expected_output_height ||
        output_width != expected_output_width
    ) {
        throw std::invalid_argument(
            "Las dimensiones de salida no coinciden con "
            "input, kernel, stride y padding"
        );
    }

    /* Calculo de la cantidad de datos
       Calculamos cuántos elementos float contiene la entrada y cuántos
       elementos tendrá la salida.
     
       El orden de las dimensiones es:

       batch × channels × height × width
    
       Estos tamaños son necesarios posteriormente para reservar los buffers
       de memoria administrados por XRT.
    */
    const std::size_t input_size =
        static_cast<std::size_t>(batch_size)
        * static_cast<std::size_t>(input_channels)
        * static_cast<std::size_t>(input_height)
        * static_cast<std::size_t>(input_width);

    const std::size_t output_size =
        static_cast<std::size_t>(batch_size)
        * static_cast<std::size_t>(input_channels)
        * static_cast<std::size_t>(output_height)
        * static_cast<std::size_t>(output_width);


    // Se convierte la cantidad de elementos a bytes, XRT necesita conocer cuánto espacio de memoria debe reservar
    const std::size_t input_bytes =
        input_size * sizeof(float);

    const std::size_t output_bytes =
        output_size * sizeof(float);

    /*
    const std::string binary_file =
        "/lib/firmware/xilinx/avgpool2d/avgpool2d.xclbin";

    xrt::device device(0);

    auto uuid = device.load_xclbin(
        binary_file
    );

    xrt::kernel kernel(
        device,
        uuid,
        "avg_pool2d_forward"
    );
    */


    
    

    
    /* Implementación con runtime
     * Se usa el runtime que ya carga el XLCBIN y crea el dispositivo
     */   
    auto& runtime = FpgaRuntime::instance();

    xrt::device& device = runtime.device();

    // Aquí conectamos el código del host con el kernel implementado en FPGA.
    xrt::kernel kernel(
        device,
        runtime.uuid(),
        "avg_pool2d_forward"
    );

    // Esto crea los buffers 
    xrt::bo input_bo(
        device,
        input_bytes,
        kernel.group_id(0)     // kernel.group_id(0) indica el grupo de memoria asociado al primer argumento del kernel.
    );

    // Buffer donde el kernel FPGA escribirá el resultado.
    // kernel.group_id(1) corresponde al segundo argumento del kernel.
    xrt::bo output_bo(
        device,
        output_bytes,
        kernel.group_id(1)
    );
    
    // map () permite obtener desde el host un puntero para acceder al contenido de los buffers.
    float* input_map =
        input_bo.map<float*>();

    float* output_map =
        output_bo.map<float*>();

    //copia los datos que orginalmente estaban en "input" hacia el buffer administrado por XRT.
    std::copy_n(
        input,
        input_size,
        input_map
    );


    /*
    // sincronización del host -> device
    // se le indica a XRT que el contenido del buffer debe estar disponible para la FPGA.
    input_bo.sync(
        XCL_BO_SYNC_BO_TO_DEVICE
    );

    // lanzamiento del kernel FPGA, se le pasan los argumentos que necesita para ejecutar la operación de AvgPool2d.
    auto run = kernel(
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

    run.wait(); // se espera hasta que la ejecución del kernel haya terminado

    // ahora se sincroniza device -> host, se traen los datos producidos por el kernel
    output_bo.sync(
        XCL_BO_SYNC_BO_FROM_DEVICE
    );

    // para finalizar se copia el resultado desde el buffer administrado por XRT hacia el puntero 
    // "output" que recibio originalmente esta función a partir de aquí el resultado vuelve a estar 
    // disponible para el código que llamó a avgpool2d_fpga_forward.
    std::copy_n(
        output_map,
        output_size,
        output
    );
}

*/



    input_bo.sync(
        XCL_BO_SYNC_BO_TO_DEVICE
    );


    
    auto run = kernel(
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


    
    run.wait();

    

    
    output_bo.sync(
        XCL_BO_SYNC_BO_FROM_DEVICE
    );


    
    std::copy_n(
        output_map,
        output_size,
        output
    );


    
}