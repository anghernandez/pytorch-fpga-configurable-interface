#include "../Pointwise/SW/pointwise_fpga.hpp"
#include "../runtime/fpga_runtime.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <xrt/xrt/xrt_kernel.h>

int main()
{
    try {
        auto& runtime = FpgaRuntime::instance();

        xrt::device& device = runtime.device();
        const xrt::uuid& uuid = runtime.uuid();

        xrt::kernel kernel(
            device,
            uuid,
            "pointwise_conv2d_forward"
        );

        // --------------------------------------------------------
        // Dimensiones
        // --------------------------------------------------------

        const int batch_size = 1;
        const int input_channels = 2;
        const int input_height = 2;
        const int input_width = 2;
        const int output_channels = 2;

        const bool use_bias = true;

        const std::size_t input_elements =
            static_cast<std::size_t>(batch_size)
            * input_channels
            * input_height
            * input_width;

        const std::size_t output_elements =
            static_cast<std::size_t>(batch_size)
            * output_channels
            * input_height
            * input_width;

        const std::size_t input_bytes =
            input_elements * sizeof(float);

        const std::size_t output_bytes =
            output_elements * sizeof(float);

        // --------------------------------------------------------
        // Datos de prueba
        // --------------------------------------------------------

        std::vector<float> input = {
            1.0f, 2.0f,
            3.0f, 4.0f,

            5.0f, 6.0f,
            7.0f, 8.0f
        };

        std::vector<float> weight = {
            1.0f, 2.0f,
            3.0f, 4.0f
        };

        std::vector<float> bias = {
            1.0f,
            2.0f
        };

        // --------------------------------------------------------
        // Tamaños de weight y bias
        // --------------------------------------------------------

        const std::size_t weight_elements =
            static_cast<std::size_t>(output_channels)
            * input_channels;

        const std::size_t weight_bytes =
            weight_elements * sizeof(float);

        const std::size_t bias_elements =
            static_cast<std::size_t>(output_channels);

        const std::size_t bias_bytes =
            bias_elements * sizeof(float);

        // --------------------------------------------------------
        // Adquirir buffers
        // --------------------------------------------------------

        FpgaBuffer input_buffer =
            runtime.acquire_buffer(
                input_bytes,
                kernel.group_id(0)
            );

        FpgaBuffer weight_buffer =
            runtime.acquire_buffer(
                weight_bytes,
                kernel.group_id(1)
            );

        FpgaBuffer bias_buffer =
            runtime.acquire_buffer(
                bias_bytes,
                kernel.group_id(2)
            );

        FpgaBuffer output_buffer =
            runtime.acquire_buffer(
                output_bytes,
                kernel.group_id(3)
            );

        // --------------------------------------------------------
        // Cargar datos al FPGA
        // --------------------------------------------------------

        runtime.upload_buffer(
            input_buffer,
            input.data(),
            input_bytes
        );

        runtime.upload_buffer(
            weight_buffer,
            weight.data(),
            weight_bytes
        );

        runtime.upload_buffer(
            bias_buffer,
            bias.data(),
            bias_bytes
        );

        // --------------------------------------------------------
        // Ejecutar Pointwise
        // --------------------------------------------------------

        pointwise_fpga_forward_buffer(
            input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer,
            batch_size,
            input_channels,
            input_height,
            input_width,
            output_channels,
            use_bias
        );

        // --------------------------------------------------------
        // Solo para verificar el resultado
        // --------------------------------------------------------

        output_buffer.bo.sync(
            XCL_BO_SYNC_BO_FROM_DEVICE
        );

        float* output_map =
            output_buffer.bo.map<float*>();

        // --------------------------------------------------------
        // Resultado esperado
        // --------------------------------------------------------

        std::vector<float> expected = {
             12.0f, 15.0f,
             18.0f, 21.0f,

             25.0f, 32.0f,
             39.0f, 46.0f
        };

        bool pass = true;

        for (std::size_t i = 0; i < output_elements; ++i) {

            float error =
                std::abs(output_map[i] - expected[i]);

            std::cout
                << "output[" << i << "] = "
                << output_map[i]
                << " | esperado = "
                << expected[i]
                << " | error = "
                << error
                << std::endl;

            if (error > 1e-5f) {
                pass = false;
            }
        }

        // --------------------------------------------------------
        // Liberar buffers
        // --------------------------------------------------------

        runtime.release_buffer(input_buffer);
        runtime.release_buffer(weight_buffer);
        runtime.release_buffer(bias_buffer);
        runtime.release_buffer(output_buffer);

        if (pass) {
            std::cout << "\nRESULTADO: PASS\n";
            return 0;
        }

        std::cout << "\nRESULTADO: FAIL\n";
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr
            << "ERROR: "
            << e.what()
            << std::endl;

        return 1;
    }
}