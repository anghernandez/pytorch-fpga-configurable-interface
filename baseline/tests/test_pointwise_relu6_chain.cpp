#include "../Pointwise/SW/pointwise_fpga.hpp"
#include "../ReLU6/SW/relu6_fpga.hpp"
#include "../runtime/fpga_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include <xrt/xrt/xrt_kernel.h>

int main()
{
    try {
        auto& runtime = FpgaRuntime::instance();

        xrt::device& device = runtime.device();
        const xrt::uuid& uuid = runtime.uuid();

        // ========================================================
        // Kernels
        // ========================================================

        xrt::kernel pointwise_kernel(
            device,
            uuid,
            "pointwise_conv2d_forward"
        );

        xrt::kernel relu6_kernel(
            device,
            uuid,
            "relu6_forward"
        );

        // ========================================================
        // Dimensiones
        // ========================================================

        const int batch_size = 1;
        const int input_channels = 2;
        const int input_height = 2;
        const int input_width = 2;
        const int output_channels = 2;

        const int size =
            batch_size *
            output_channels *
            input_height *
            input_width;

        const std::size_t input_elements =
            static_cast<std::size_t>(
                batch_size *
                input_channels *
                input_height *
                input_width
            );

        const std::size_t output_elements =
            static_cast<std::size_t>(size);

        // ========================================================
        // Datos de entrada
        // ========================================================

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

        // ========================================================
        // REFERENCIA CPU
        //
        // Primero Pointwise
        // Después ReLU6
        // ========================================================

        std::vector<float> expected_pointwise(
            output_elements
        );

        std::vector<float> expected_final(
            output_elements
        );

        /*
         * Layout utilizado por el kernel:
         *
         * posición 0 -> input[0], input[4]
         * posición 1 -> input[1], input[5]
         * posición 2 -> input[2], input[6]
         * posición 3 -> input[3], input[7]
         */

        for (int position = 0; position < 4; ++position) {

            float input_c0 =
                input[position];

            float input_c1 =
                input[4 + position];

            // Output channel 0
            expected_pointwise[position] =
                input_c0 * weight[0]
                + input_c1 * weight[1]
                + bias[0];

            // Output channel 1
            expected_pointwise[4 + position] =
                input_c0 * weight[2]
                + input_c1 * weight[3]
                + bias[1];
        }

        // Aplicar ReLU6 sobre la salida Pointwise
        for (std::size_t i = 0; i < output_elements; ++i) {

            expected_final[i] =
                std::min(
                    std::max(
                        expected_pointwise[i],
                        0.0f
                    ),
                    6.0f
                );
        }

        // ========================================================
        // BUFFER DE ENTRADA
        // ========================================================

        FpgaBuffer input_buffer =
            runtime.acquire_buffer(
                input_elements * sizeof(float),
                pointwise_kernel.group_id(0)
            );

        // ========================================================
        // BUFFER DE SALIDA POINTWISE
        //
        // Este mismo BO será entregado a ReLU6.
        // ========================================================

        FpgaBuffer pointwise_output_buffer =
            runtime.acquire_buffer(
                output_elements * sizeof(float),
                pointwise_kernel.group_id(3)
            );

        // ========================================================
        // BUFFER DE SALIDA RELU6
        // ========================================================

        FpgaBuffer relu6_output_buffer =
            runtime.acquire_buffer(
                output_elements * sizeof(float),
                relu6_kernel.group_id(1)
            );

        std::cout << "\n==============================\n";
        std::cout << "CONFIGURACION DE BUFFERS\n";
        std::cout << "==============================\n";

        std::cout
            << "Input group       = "
            << input_buffer.memory_group
            << std::endl;

        std::cout
            << "Pointwise output  = "
            << pointwise_output_buffer.memory_group
            << std::endl;

        std::cout
            << "ReLU6 input       = "
            << relu6_kernel.group_id(0)
            << std::endl;

        std::cout
            << "ReLU6 output      = "
            << relu6_output_buffer.memory_group
            << std::endl;

        // ========================================================
        // CPU → FPGA
        //
        // ÚNICA transferencia de entrada.
        // ========================================================

        float* input_map =
            input_buffer.bo.map<float*>();

        std::copy(
            input.begin(),
            input.end(),
            input_map
        );

        input_buffer.bo.sync(
            XCL_BO_SYNC_BO_TO_DEVICE
        );

        // ========================================================
        // POINTWISE
        //
        // Input:  input_buffer
        // Output: pointwise_output_buffer
        // ========================================================

        std::cout << "\n==============================\n";
        std::cout << "KERNEL 1: POINTWISE\n";
        std::cout << "==============================\n";

        pointwise_fpga_forward_buffer(
            input_buffer,
            weight.data(),
            bias.data(),
            pointwise_output_buffer,
            batch_size,
            input_channels,
            input_height,
            input_width,
            output_channels,
            true
        );

        // IMPORTANTE:
        //
        // NO hacemos:
        //
        // pointwise_output_buffer.bo.sync(
        //     XCL_BO_SYNC_BO_FROM_DEVICE
        // );
        //
        // El resultado permanece en FPGA.

        std::cout
            << "Pointwise terminado."
            << std::endl;

        // ========================================================
        // RELU6
        //
        // Consume DIRECTAMENTE el BO anterior.
        // ========================================================

        std::cout << "\n==============================\n";
        std::cout << "KERNEL 2: RELU6\n";
        std::cout << "==============================\n";

        relu6_fpga_forward_buffer(
            pointwise_output_buffer,
            relu6_output_buffer,
            size
        );

        std::cout
            << "ReLU6 terminado."
            << std::endl;

        // ========================================================
        // FPGA → CPU
        //
        // ÚNICA transferencia de salida.
        // ========================================================

        relu6_output_buffer.bo.sync(
            XCL_BO_SYNC_BO_FROM_DEVICE
        );

        float* final_map =
            relu6_output_buffer.bo.map<float*>();

        // ========================================================
        // COMPARACION
        // ========================================================

        std::cout << "\n==============================\n";
        std::cout << "COMPARACION FINAL\n";
        std::cout << "==============================\n";

        bool pass = true;

        float max_error = 0.0f;

        for (std::size_t i = 0; i < output_elements; ++i) {

            float error =
                std::abs(
                    final_map[i]
                    - expected_final[i]
                );

            max_error =
                std::max(
                    max_error,
                    error
                );

            std::cout
                << "output[" << i << "] = "
                << final_map[i]
                << " | esperado = "
                << expected_final[i]
                << " | error = "
                << error
                << std::endl;

            if (error > 1e-5f) {
                pass = false;
            }
        }

        std::cout
            << "\nMax error = "
            << max_error
            << std::endl;

        // ========================================================
        // LIBERAR
        // ========================================================

        runtime.release_buffer(input_buffer);
        runtime.release_buffer(pointwise_output_buffer);
        runtime.release_buffer(relu6_output_buffer);

        // ========================================================
        // RESULTADO
        // ========================================================

        if (pass) {
            std::cout
                << "\nRESULTADO: PASS\n";

            std::cout
                << "Pointwise -> ReLU6 "
                   "ejecutado correctamente.\n";

            return 0;
        }

        std::cout
            << "\nRESULTADO: FAIL\n";

        return 1;
    }
    catch (const std::exception& e) {

        std::cerr
            << "\nERROR: "
            << e.what()
            << std::endl;

        return 1;
    }
}