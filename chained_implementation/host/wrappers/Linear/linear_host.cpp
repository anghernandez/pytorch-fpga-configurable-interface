#include <iostream>
#include <vector>
#include <cmath>

#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>
#include <xrt/xrt/xrt_bo.h>

int main()
{
    const int batch_size = 64;
    const int in_features = 1280;
    const int out_features = 1000;
    const bool use_bias = true;

    const int input_size =
        batch_size * in_features;

    const int weight_size =
        out_features * in_features;

    const int bias_size =
        out_features;

    const int output_size =
        batch_size * out_features;

    const size_t input_bytes =
        input_size * sizeof(float);

    const size_t weight_bytes =
        weight_size * sizeof(float);

    const size_t bias_bytes =
        bias_size * sizeof(float);

    const size_t output_bytes =
        output_size * sizeof(float);


    // ---------------------------------------------------------
    // Datos de prueba
    // ---------------------------------------------------------

    std::vector<float> input(input_size);
    std::vector<float> weight(weight_size);
    std::vector<float> bias(bias_size);
    std::vector<float> output_ref(output_size);

    for (int i = 0; i < input_size; i++) {
        input[i] =
            static_cast<float>((i % 17) - 8) * 0.05f;
    }

    for (int i = 0; i < weight_size; i++) {
        weight[i] =
            static_cast<float>((i % 13) - 6) * 0.02f;
    }

    for (int i = 0; i < bias_size; i++) {
        bias[i] =
            static_cast<float>((i % 7) - 3) * 0.01f;
    }


    // ---------------------------------------------------------
    // Referencia software
    // ---------------------------------------------------------

    for (int sample_index = 0;
         sample_index < batch_size;
         sample_index++) {

        for (int output_index = 0;
             output_index < out_features;
             output_index++) {

            float accumulated_value = 0.0f;

            for (int input_index = 0;
                 input_index < in_features;
                 input_index++) {

                const int input_position =
                    sample_index * in_features
                    + input_index;

                const int weight_position =
                    output_index * in_features
                    + input_index;

                accumulated_value +=
                    input[input_position]
                    * weight[weight_position];
            }

            if (use_bias) {
                accumulated_value +=
                    bias[output_index];
            }

            const int output_position =
                sample_index * out_features
                + output_index;

            output_ref[output_position] =
                accumulated_value;
        }
    }


    // ---------------------------------------------------------
    // Abrir dispositivo y cargar xclbin
    // ---------------------------------------------------------

    std::cout << "Abriendo dispositivo..." << std::endl;

    xrt::device device(0);

    std::cout << "Cargando xclbin..." << std::endl;

    auto uuid = device.load_xclbin(
        "/lib/firmware/xilinx/linear/linear.xclbin"
    );


    // ---------------------------------------------------------
    // Abrir kernel
    // ---------------------------------------------------------

    std::cout
        << "Abriendo kernel linear_forward..."
        << std::endl;

    xrt::kernel kernel(
        device,
        uuid,
        "linear_forward"
    );


    // ---------------------------------------------------------
    // Crear buffers
    // ---------------------------------------------------------

    std::cout << "Creando buffers..." << std::endl;

    xrt::bo input_bo(
        device,
        input_bytes,
        kernel.group_id(0)
    );

    xrt::bo weight_bo(
        device,
        weight_bytes,
        kernel.group_id(1)
    );

    xrt::bo bias_bo(
        device,
        bias_bytes,
        kernel.group_id(2)
    );

    xrt::bo output_bo(
        device,
        output_bytes,
        kernel.group_id(3)
    );


    // ---------------------------------------------------------
    // Mapear buffers
    // ---------------------------------------------------------

    float* input_map =
        input_bo.map<float*>();

    float* weight_map =
        weight_bo.map<float*>();

    float* bias_map =
        bias_bo.map<float*>();

    float* output_map =
        output_bo.map<float*>();


    // ---------------------------------------------------------
    // Copiar datos
    // ---------------------------------------------------------

    for (int i = 0; i < input_size; i++) {
        input_map[i] = input[i];
    }

    for (int i = 0; i < weight_size; i++) {
        weight_map[i] = weight[i];
    }

    for (int i = 0; i < bias_size; i++) {
        bias_map[i] = bias[i];
    }

    for (int i = 0; i < output_size; i++) {
        output_map[i] = 0.0f;
    }


    // ---------------------------------------------------------
    // Host -> FPGA
    // ---------------------------------------------------------

    std::cout
        << "Enviando datos a FPGA..."
        << std::endl;

    input_bo.sync(
        XCL_BO_SYNC_BO_TO_DEVICE
    );

    weight_bo.sync(
        XCL_BO_SYNC_BO_TO_DEVICE
    );

    bias_bo.sync(
        XCL_BO_SYNC_BO_TO_DEVICE
    );


    // ---------------------------------------------------------
    // Ejecutar kernel
    // ---------------------------------------------------------

    std::cout
        << "Ejecutando linear_forward..."
        << std::endl;


    std::cout
    << "Parametros enviados:"
    << " batch_size=" << batch_size
    << " in_features=" << in_features
    << " out_features=" << out_features
    << " use_bias=" << use_bias
    << std::endl;


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

    run.wait();


    // ---------------------------------------------------------
    // FPGA -> Host
    // ---------------------------------------------------------

    std::cout
        << "Leyendo resultados..."
        << std::endl;

    output_bo.sync(
        XCL_BO_SYNC_BO_FROM_DEVICE
    );


    // ---------------------------------------------------------
    // Mostrar primeras 10 salidas
    // ---------------------------------------------------------

    std::cout
        << "\nPrimeras 10 salidas:\n";

    for (int i = 0; i < 10; i++) {

        const double error =
            std::fabs(
                static_cast<double>(output_map[i])
                -
                static_cast<double>(output_ref[i])
            );

        std::cout
            << "output[" << i << "]"
            << "  HW = " << output_map[i]
            << "  REF = " << output_ref[i]
            << "  error = " << error
            << std::endl;
    }


    // ---------------------------------------------------------
    // Calcular error
    // ---------------------------------------------------------

    double squared_error_sum = 0.0;
    double max_abs_error = 0.0;

    for (int i = 0; i < output_size; i++) {

        const double error =
            static_cast<double>(output_map[i])
            -
            static_cast<double>(output_ref[i]);

        squared_error_sum +=
            error * error;

        const double abs_error =
            std::fabs(error);

        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
        }
    }

    const double rmse =
        std::sqrt(
            squared_error_sum
            /
            static_cast<double>(output_size)
        );


    // ---------------------------------------------------------
    // Resultado
    // ---------------------------------------------------------

    std::cout
        << "\nRMSE = "
        << rmse
        << std::endl;

    std::cout
        << "Max absolute error = "
        << max_abs_error
        << std::endl;

    if (rmse <= 1e-3) {

        std::cout
            << "\nTEST PASSED"
            << std::endl;

        return 0;
    }

    std::cout
        << "\nTEST FAILED"
        << std::endl;

    return 1;
}
