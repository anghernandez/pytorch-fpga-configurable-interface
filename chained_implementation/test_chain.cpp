#include "fpga_runtime.hpp"
#include "relu6_fpga.hpp"
#include "tanh_fpga.hpp"
#include "linear_fpga.hpp"
#include "conv2d_fpga.hpp"
#include "avgpool2d_fpga.hpp"
#include "pointwise_fpga.hpp"
#include "depthwise_fpga.hpp"
#include "batchnorm2d_fpga.hpp"
#include "layeradd_fpga.hpp"

#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_kernel.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

int main()
{
    try {
        // ============================================================
        // 1. RUNTIME COMPARTIDO: UN DISPOSITIVO Y UN XCLBIN
        // ============================================================
        auto& runtime = FpgaRuntime::instance();
        xrt::device& device = runtime.device();
        const auto& uuid = runtime.uuid();

        xrt::kernel relu6(device, uuid, "relu6_forward");
        xrt::kernel tanh(device, uuid, "tanh_forward");
        xrt::kernel linear(device, uuid, "linear_forward");
        xrt::kernel conv2d(device, uuid, "conv2d_forward");
        xrt::kernel avgpool(device, uuid, "avg_pool2d_forward");
        xrt::kernel pointwise(device, uuid, "pointwise_conv2d_forward");
        xrt::kernel depthwise(device, uuid, "depthwise_conv2d_forward");
        xrt::kernel batchnorm(device, uuid, "batchnorm2d_forward");
        xrt::kernel layeradd(device, uuid, "layer_add_forward");

        // ============================================================
        // 2. PRUEBA EXISTENTE: RELU6 -> TANH
        // ============================================================
        {
            constexpr int SIZE = 16;
            constexpr std::size_t BYTES = SIZE * sizeof(float);

            xrt::bo buffer_A(device, BYTES, relu6.group_id(0));
            xrt::bo buffer_B(device, BYTES, relu6.group_id(1));
            xrt::bo buffer_C(device, BYTES, tanh.group_id(1));

            auto* input = buffer_A.map<float*>();
            std::vector<float> reference(SIZE);

            for (int i = 0; i < SIZE; ++i) {
                input[i] = static_cast<float>(i) - 8.0f;
                const float activated = std::min(6.0f, std::max(0.0f, input[i]));
                reference[i] = std::tanh(activated);
            }
            buffer_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // Buffer B se consume directamente; nunca se sincroniza a CPU.
            relu6_fpga_forward(buffer_A, buffer_B, SIZE);
            tanh_fpga_forward(buffer_B, buffer_C, SIZE);

            buffer_C.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            auto* output = buffer_C.map<float*>();
            double sum_sq = 0.0;
            for (int i = 0; i < SIZE; ++i) {
                double difference = static_cast<double>(output[i]) - reference[i];
                sum_sq += difference * difference;
            }
            double rmse = std::sqrt(sum_sq / SIZE);
            std::cout << "[ReLU6 -> Tanh] RMSE: " << rmse
                      << " | " << (rmse <= 1e-3 ? "PASS" : "FAIL") << '\n';
            if (rmse > 1e-3) return 1;
        }

        // ============================================================
        // 3. PRUEBA NUEVA: LINEAR CON PESOS Y BIAS REUTILIZADOS
        // ============================================================
        {
            constexpr int BATCH = 2;
            constexpr int IN_FEATURES = 4;
            constexpr int OUT_FEATURES = 3;
            constexpr int EXECUTIONS = 2;
            constexpr bool USE_BIAS = true;

            constexpr std::size_t INPUT_COUNT = BATCH * IN_FEATURES;
            constexpr std::size_t WEIGHT_COUNT = OUT_FEATURES * IN_FEATURES;
            constexpr std::size_t BIAS_COUNT = OUT_FEATURES;
            constexpr std::size_t OUTPUT_COUNT = BATCH * OUT_FEATURES;

            xrt::bo input_bo(device, INPUT_COUNT * sizeof(float), linear.group_id(0));
            xrt::bo weight_bo(device, WEIGHT_COUNT * sizeof(float), linear.group_id(1));
            xrt::bo bias_bo(device, BIAS_COUNT * sizeof(float), linear.group_id(2));
            xrt::bo output_bo(device, OUTPUT_COUNT * sizeof(float), linear.group_id(3));

            auto* input = input_bo.map<float*>();
            auto* weight = weight_bo.map<float*>();
            auto* bias = bias_bo.map<float*>();

            for (std::size_t i = 0; i < INPUT_COUNT; ++i)
                input[i] = (static_cast<float>(i) + 1.0f) * 0.1f;
            for (std::size_t i = 0; i < WEIGHT_COUNT; ++i)
                weight[i] = (static_cast<float>(i) + 1.0f) * 0.05f;
            for (std::size_t i = 0; i < BIAS_COUNT; ++i)
                bias[i] = static_cast<float>(i) * 0.1f;

            std::vector<float> reference(OUTPUT_COUNT);
            for (int b = 0; b < BATCH; ++b) {
                for (int o = 0; o < OUT_FEATURES; ++o) {
                    float value = bias[o];
                    for (int i = 0; i < IN_FEATURES; ++i)
                        value += input[b * IN_FEATURES + i] * weight[o * IN_FEATURES + i];
                    reference[b * OUT_FEATURES + o] = value;
                }
            }

            // Transferir entrada y parámetros una sola vez.
            input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bias_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            for (int execution = 1; execution <= EXECUTIONS; ++execution) {
                linear_fpga_forward(input_bo, weight_bo, bias_bo, output_bo,
                                    BATCH, IN_FEATURES, OUT_FEATURES, USE_BIAS);
                // Solo para validar: recuperar salida tras cada ejecución.
                output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                auto* output = output_bo.map<float*>();

                double sum_sq = 0.0;
                double max_error = 0.0;
                for (std::size_t i = 0; i < OUTPUT_COUNT; ++i) {
                    double error = std::abs(static_cast<double>(output[i]) - reference[i]);
                    sum_sq += error * error;
                    max_error = std::max(max_error, error);
                }
                double rmse = std::sqrt(sum_sq / OUTPUT_COUNT);
                std::cout << "[Linear ejecucion " << execution << "] RMSE: " << rmse
                          << " | error maximo: " << max_error
                          << " | " << (rmse <= 1e-3 ? "PASS" : "FAIL") << '\n';
                if (rmse > 1e-3) return 1;
            }
        }

        // ============================================================
        // 4. PRUEBA CONV2D: BUFFERS EXTERNOS Y REFERENCIA CPU
        // ============================================================
        {
            constexpr int B = 1, CI = 1, H = 4, W = 4;
            constexpr int CO = 2, KH = 3, KW = 3;
            constexpr int SH = 1, SW = 1, PH = 0, PW = 0;
            constexpr int OH = 2, OW = 2;
            constexpr std::size_t INPUT_COUNT = B * CI * H * W;
            constexpr std::size_t WEIGHT_COUNT = CO * CI * KH * KW;
            constexpr std::size_t BIAS_COUNT = CO;
            constexpr std::size_t OUTPUT_COUNT = B * CO * OH * OW;

            xrt::bo input_bo(device, INPUT_COUNT * sizeof(float), conv2d.group_id(0));
            xrt::bo weight_bo(device, WEIGHT_COUNT * sizeof(float), conv2d.group_id(1));
            xrt::bo bias_bo(device, BIAS_COUNT * sizeof(float), conv2d.group_id(2));
            xrt::bo output_bo(device, OUTPUT_COUNT * sizeof(float), conv2d.group_id(3));

            auto* input = input_bo.map<float*>();
            auto* weight = weight_bo.map<float*>();
            auto* bias = bias_bo.map<float*>();
            for (std::size_t i = 0; i < INPUT_COUNT; ++i)
                input[i] = static_cast<float>(i + 1) * 0.1f;
            for (std::size_t i = 0; i < WEIGHT_COUNT; ++i)
                weight[i] = static_cast<float>(i + 1) * 0.02f;
            for (int o = 0; o < CO; ++o)
                bias[o] = 0.1f * static_cast<float>(o + 1);

            // Referencia: convolución NCHW, pesos [CO, CI, KH, KW].
            std::vector<float> reference(OUTPUT_COUNT, 0.0f);
            for (int b = 0; b < B; ++b)
                for (int o = 0; o < CO; ++o)
                    for (int y = 0; y < OH; ++y)
                        for (int x = 0; x < OW; ++x) {
                            float sum = bias[o];
                            for (int c = 0; c < CI; ++c)
                                for (int ky = 0; ky < KH; ++ky)
                                    for (int kx = 0; kx < KW; ++kx) {
                                        const int iy = y * SH + ky - PH;
                                        const int ix = x * SW + kx - PW;
                                        if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                                            sum += input[((b * CI + c) * H + iy) * W + ix]
                                                * weight[((o * CI + c) * KH + ky) * KW + kx];
                                    }
                            reference[((b * CO + o) * OH + y) * OW + x] = sum;
                        }

            input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bias_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            conv2d_fpga_forward(input_bo, weight_bo, bias_bo, output_bo,
                                B, CI, H, W, CO, OH, OW, KH, KW,
                                SH, SW, PH, PW, true);
            // Recuperación al final de esta prueba, no dentro del wrapper.
            output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            auto* output = output_bo.map<float*>();
            double sum_sq = 0.0;
            double max_error = 0.0;
            for (std::size_t i = 0; i < OUTPUT_COUNT; ++i) {
                const double error = std::abs(static_cast<double>(output[i]) - reference[i]);
                sum_sq += error * error;
                max_error = std::max(max_error, error);
            }
            const double rmse = std::sqrt(sum_sq / OUTPUT_COUNT);
            std::cout << "[Conv2D] RMSE: " << rmse
                      << " | error maximo: " << max_error
                      << " | " << (rmse <= 1e-3 ? "PASS" : "FAIL") << '\n';
            if (rmse > 1e-3) return 1;
        }


        // ============================================================
        // 5. PRUEBA ENCADENADA: CONV2D -> AVGPOOL2D
        // ============================================================
        // Ambas capas son independientes. El banco de pruebas decide
        // qué buffer usar como salida de Conv2D y entrada de AvgPool2D.
        {
            constexpr int B = 1, CI = 1, H = 4, W = 4;
            constexpr int CO = 2, KH = 3, KW = 3;
            constexpr int CH = 2, CW = 2; // salida de Conv2D
            constexpr int PH = 0, PW = 0, STRIDE = 1;
            constexpr int POOL_K = 2, POOL_STRIDE = 2;
            constexpr int OH = 1, OW = 1; // salida de AvgPool2D
            constexpr std::size_t INPUT_COUNT = B * CI * H * W;
            constexpr std::size_t WEIGHT_COUNT = CO * CI * KH * KW;
            constexpr std::size_t BIAS_COUNT = CO;
            constexpr std::size_t CONV_COUNT = B * CO * CH * CW;
            constexpr std::size_t OUTPUT_COUNT = B * CO * OH * OW;

            xrt::bo input_bo(device, INPUT_COUNT * sizeof(float), conv2d.group_id(0));
            xrt::bo weight_bo(device, WEIGHT_COUNT * sizeof(float), conv2d.group_id(1));
            xrt::bo bias_bo(device, BIAS_COUNT * sizeof(float), conv2d.group_id(2));
            xrt::bo intermediate_bo(device, CONV_COUNT * sizeof(float), conv2d.group_id(3));
            xrt::bo output_bo(device, OUTPUT_COUNT * sizeof(float), avgpool.group_id(1));

            auto* input = input_bo.map<float*>();
            auto* weight = weight_bo.map<float*>();
            auto* bias = bias_bo.map<float*>();
            for (std::size_t i = 0; i < INPUT_COUNT; ++i)
                input[i] = (static_cast<float>(i) + 1.0f) * 0.1f;
            for (std::size_t i = 0; i < WEIGHT_COUNT; ++i)
                weight[i] = (static_cast<float>(i) + 1.0f) * 0.02f;
            for (int o = 0; o < CO; ++o)
                bias[o] = (static_cast<float>(o) + 1.0f) * 0.1f;

            // Referencia CPU: Conv2D (NCHW) seguida de promedio 2x2.
            std::vector<float> conv_ref(CONV_COUNT, 0.0f);
            std::vector<float> reference(OUTPUT_COUNT, 0.0f);
            for (int o = 0; o < CO; ++o)
                for (int y = 0; y < CH; ++y)
                    for (int x = 0; x < CW; ++x) {
                        float sum = bias[o];
                        for (int ky = 0; ky < KH; ++ky)
                            for (int kx = 0; kx < KW; ++kx)
                                sum += input[(y + ky) * W + (x + kx)]
                                     * weight[o * KH * KW + ky * KW + kx];
                        conv_ref[o * CH * CW + y * CW + x] = sum;
                    }
            for (int o = 0; o < CO; ++o) {
                float sum = 0.0f;
                for (int i = 0; i < CH * CW; ++i)
                    sum += conv_ref[o * CH * CW + i];
                reference[o] = sum / static_cast<float>(POOL_K * POOL_K);
            }

            input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bias_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            conv2d_fpga_forward(input_bo, weight_bo, bias_bo, intermediate_bo,
                                B, CI, H, W, CO, CH, CW, KH, KW,
                                STRIDE, STRIDE, PH, PW, true);

            // NO sincronizamos intermediate_bo hacia la CPU.
            avgpool2d_fpga_forward(intermediate_bo, output_bo,
                                   B, CO, CH, CW, OH, OW,
                                   POOL_K, POOL_K, POOL_STRIDE, POOL_STRIDE,
                                   0, 0, true, false, 0);

            output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            auto* output = output_bo.map<float*>();
            double sum_sq = 0.0;
            double max_error = 0.0;
            for (std::size_t i = 0; i < OUTPUT_COUNT; ++i) {
                double error = std::abs(static_cast<double>(output[i]) - reference[i]);
                sum_sq += error * error;
                max_error = std::max(max_error, error);
            }
            const double rmse = std::sqrt(sum_sq / OUTPUT_COUNT);
            std::cout << "[Conv2D -> AvgPool2D] RMSE: " << rmse
                      << " | error maximo: " << max_error
                      << " | " << (rmse <= 1e-3 ? "PASS" : "FAIL") << '\n';
            if (rmse > 1e-3) return 1;
        }


        // ============================================================
        // 6. POINTWISE -> RELU6 CON BUFFER INTERMEDIO COMPARTIDO
        // ============================================================
        {
            constexpr int B = 1, CI = 2, H = 2, W = 3, CO = 3;
            constexpr std::size_t NI = B * CI * H * W;
            constexpr std::size_t NW = CO * CI;
            constexpr std::size_t NB = CO;
            constexpr std::size_t NO = B * CO * H * W;
            xrt::bo input_bo(device, NI * sizeof(float), pointwise.group_id(0));
            xrt::bo weight_bo(device, NW * sizeof(float), pointwise.group_id(1));
            xrt::bo bias_bo(device, NB * sizeof(float), pointwise.group_id(2));
            xrt::bo intermediate_bo(device, NO * sizeof(float), pointwise.group_id(3));
            xrt::bo output_bo(device, NO * sizeof(float), relu6.group_id(1));
            auto* input = input_bo.map<float*>();
            auto* weight = weight_bo.map<float*>();
            auto* bias = bias_bo.map<float*>();
            for (std::size_t i = 0; i < NI; ++i)
                input[i] = (static_cast<float>(i) - 4.0f) * 0.3f;
            for (std::size_t i = 0; i < NW; ++i)
                weight[i] = (static_cast<float>(i) - 2.0f) * 0.2f;
            for (int o = 0; o < CO; ++o)
                bias[o] = 0.1f * (static_cast<float>(o) - 1.0f);

            // Referencia CPU, NCHW: pointwise 1x1 y ReLU6.
            std::vector<float> reference(NO);
            for (int b = 0; b < B; ++b)
                for (int o = 0; o < CO; ++o)
                    for (int y = 0; y < H; ++y)
                        for (int x = 0; x < W; ++x) {
                            float acc = bias[o];
                            for (int i = 0; i < CI; ++i)
                                acc += input[((b * CI + i) * H + y) * W + x]
                                     * weight[o * CI + i];
                            reference[((b * CO + o) * H + y) * W + x] =
                                std::min(6.0f, std::max(0.0f, acc));
                        }

            input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bias_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            pointwise_fpga_forward(input_bo, weight_bo, bias_bo, intermediate_bo,
                                   B, CI, H, W, CO, true);
            // Sin sincronizar intermediate_bo hacia la CPU.
            relu6_fpga_forward(intermediate_bo, output_bo, static_cast<int>(NO));
            output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            auto* result = output_bo.map<float*>();
            double sum_sq = 0.0, max_error = 0.0;
            for (std::size_t i = 0; i < NO; ++i) {
                const double error = std::abs(static_cast<double>(result[i]) - reference[i]);
                sum_sq += error * error;
                max_error = std::max(max_error, error);
            }
            const double rmse = std::sqrt(sum_sq / NO);
            std::cout << "[Pointwise -> ReLU6] RMSE: " << rmse
                      << " | error maximo: " << max_error
                      << " | " << (rmse <= 1e-3 ? "PASS" : "FAIL") << '\n';
            if (rmse > 1e-3) return 1;
        }

        // ============================================================
        // 7. POINTWISE -> DEPTHWISE -> RELU6 (TRES KERNELS)
        // ============================================================
        {
            constexpr int B = 1, CI = 2, CO = 2;
            constexpr int H = 3, W = 3, KH = 3, KW = 3;
            constexpr int SH = 1, SW = 1, PH = 1, PW = 1;
            constexpr int OH = 3, OW = 3;
            constexpr std::size_t NI = B * CI * H * W;
            constexpr std::size_t NP = CO * CI;
            constexpr std::size_t ND = CO * KH * KW;
            constexpr std::size_t NO = B * CO * OH * OW;
            // Entradas, pesos y bias son independientes de los wrappers.
            xrt::bo input_bo(device, NI * sizeof(float), pointwise.group_id(0));
            xrt::bo pw_weight_bo(device, NP * sizeof(float), pointwise.group_id(1));
            xrt::bo pw_bias_bo(device, CO * sizeof(float), pointwise.group_id(2));
            xrt::bo pw_output_bo(device, NO * sizeof(float), pointwise.group_id(3));
            xrt::bo dw_weight_bo(device, ND * sizeof(float), depthwise.group_id(1));
            xrt::bo dw_bias_bo(device, CO * sizeof(float), depthwise.group_id(2));
            xrt::bo dw_output_bo(device, NO * sizeof(float), depthwise.group_id(3));
            xrt::bo output_bo(device, NO * sizeof(float), relu6.group_id(1));

            auto* input = input_bo.map<float*>();
            auto* pw_weight = pw_weight_bo.map<float*>();
            auto* pw_bias = pw_bias_bo.map<float*>();
            auto* dw_weight = dw_weight_bo.map<float*>();
            auto* dw_bias = dw_bias_bo.map<float*>();
            for (std::size_t i = 0; i < NI; ++i)
                input[i] = (static_cast<float>(i % 9) - 4.0f) * 0.2f;
            for (std::size_t i = 0; i < NP; ++i)
                pw_weight[i] = (static_cast<float>(i) - 1.0f) * 0.25f;
            for (int c = 0; c < CO; ++c) {
                pw_bias[c] = 0.05f * (c + 1);
                dw_bias[c] = -0.03f * (c + 1);
            }
            for (std::size_t i = 0; i < ND; ++i)
                dw_weight[i] = (static_cast<float>(i % 7) - 3.0f) * 0.1f;

            // Referencia CPU NCHW: pointwise 1x1, depthwise 3x3, ReLU6.
            std::vector<float> pw_ref(NO, 0.0f);
            std::vector<float> reference(NO, 0.0f);
            for (int oc = 0; oc < CO; ++oc)
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x) {
                        float sum = pw_bias[oc];
                        for (int ic = 0; ic < CI; ++ic)
                            sum += input[(ic * H + y) * W + x] * pw_weight[oc * CI + ic];
                        pw_ref[(oc * H + y) * W + x] = sum;
                    }
            for (int c = 0; c < CO; ++c)
                for (int y = 0; y < OH; ++y)
                    for (int x = 0; x < OW; ++x) {
                        float sum = dw_bias[c];
                        for (int ky = 0; ky < KH; ++ky)
                            for (int kx = 0; kx < KW; ++kx) {
                                const int iy = y * SH + ky - PH;
                                const int ix = x * SW + kx - PW;
                                if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                                    sum += pw_ref[(c * H + iy) * W + ix]
                                         * dw_weight[(c * KH + ky) * KW + kx];
                            }
                        reference[(c * OH + y) * OW + x] =
                            std::min(6.0f, std::max(0.0f, sum));
                    }

            // Sincronizacion inicial de entradas y parametros, una sola vez.
            input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            pw_weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            pw_bias_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            dw_weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            dw_bias_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            pointwise_fpga_forward(input_bo, pw_weight_bo, pw_bias_bo,
                                   pw_output_bo, B, CI, H, W, CO, true);
            // pw_output_bo no vuelve a CPU.
            depthwise_fpga_forward(pw_output_bo, dw_weight_bo, dw_bias_bo,
                                   dw_output_bo, B, CO, H, W, OH, OW,
                                   KH, KW, SH, SW, PH, PW, true);
            // dw_output_bo no vuelve a CPU.
            relu6_fpga_forward(dw_output_bo, output_bo, static_cast<int>(NO));

            output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const auto* output = output_bo.map<float*>();
            double sum_sq = 0.0, max_error = 0.0;
            for (std::size_t i = 0; i < NO; ++i) {
                const double error = std::abs(static_cast<double>(output[i]) - reference[i]);
                sum_sq += error * error;
                max_error = std::max(max_error, error);
            }
            const double rmse = std::sqrt(sum_sq / NO);
            std::cout << "[Pointwise -> Depthwise -> ReLU6] RMSE: " << rmse
                      << " | error maximo: " << max_error
                      << " | " << (rmse <= 1e-3 ? "PASS" : "FAIL") << '\n';
            if (rmse > 1e-3) return 1;
        }

        // ============================================================
        // 8. POINTWISE -> BATCHNORM2D -> RELU6
        // ============================================================
        {
            constexpr int B = 1, CI = 2, CO = 2, H = 3, W = 3;
            constexpr float EPS = 1e-5f;
            constexpr std::size_t INPUT_COUNT = B * CI * H * W;
            constexpr std::size_t WEIGHT_COUNT = CO * CI;
            constexpr std::size_t OUTPUT_COUNT = B * CO * H * W;
            constexpr std::size_t PARAM_COUNT = CO;

            xrt::bo input_bo(device, INPUT_COUNT * sizeof(float), pointwise.group_id(0));
            xrt::bo pw_weight_bo(device, WEIGHT_COUNT * sizeof(float), pointwise.group_id(1));
            xrt::bo pw_bias_bo(device, PARAM_COUNT * sizeof(float), pointwise.group_id(2));
            xrt::bo pw_output_bo(device, OUTPUT_COUNT * sizeof(float), pointwise.group_id(3));
            xrt::bo bn_weight_bo(device, PARAM_COUNT * sizeof(float), batchnorm.group_id(1));
            xrt::bo bn_bias_bo(device, PARAM_COUNT * sizeof(float), batchnorm.group_id(2));
            xrt::bo bn_mean_bo(device, PARAM_COUNT * sizeof(float), batchnorm.group_id(3));
            xrt::bo bn_var_bo(device, PARAM_COUNT * sizeof(float), batchnorm.group_id(4));
            xrt::bo bn_output_bo(device, OUTPUT_COUNT * sizeof(float), batchnorm.group_id(5));
            xrt::bo output_bo(device, OUTPUT_COUNT * sizeof(float), relu6.group_id(1));

            auto* input = input_bo.map<float*>();
            auto* pw_weight = pw_weight_bo.map<float*>();
            auto* pw_bias = pw_bias_bo.map<float*>();
            auto* gamma = bn_weight_bo.map<float*>();
            auto* beta = bn_bias_bo.map<float*>();
            auto* mean = bn_mean_bo.map<float*>();
            auto* variance = bn_var_bo.map<float*>();

            for (std::size_t i = 0; i < INPUT_COUNT; ++i)
                input[i] = (static_cast<float>(i % 9) - 4.0f) * 0.15f;
            for (std::size_t i = 0; i < WEIGHT_COUNT; ++i)
                pw_weight[i] = (static_cast<float>(i) - 1.0f) * 0.25f;
            for (int c = 0; c < CO; ++c) {
                pw_bias[c] = 0.02f * (c + 1);
                gamma[c] = 0.8f + 0.2f * c;
                beta[c] = -0.1f + 0.05f * c;
                mean[c] = 0.03f * (c + 1);
                variance[c] = 0.5f + 0.3f * c;
            }

            // Referencia CPU: convolucion 1x1, normalizacion en inferencia, ReLU6.
            std::vector<float> reference(OUTPUT_COUNT);
            for (int oc = 0; oc < CO; ++oc)
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x) {
                        float sum = pw_bias[oc];
                        for (int ic = 0; ic < CI; ++ic)
                            sum += input[(ic * H + y) * W + x] * pw_weight[oc * CI + ic];
                        const float normalized =
                            (sum - mean[oc]) / std::sqrt(variance[oc] + EPS)
                            * gamma[oc] + beta[oc];
                        reference[(oc * H + y) * W + x] =
                            std::min(6.0f, std::max(0.0f, normalized));
                    }

            // Unica carga de entradas y parametros: no se leen intermedios.
            input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            pw_weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            pw_bias_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bn_weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bn_bias_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bn_mean_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bn_var_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            pointwise_fpga_forward(input_bo, pw_weight_bo, pw_bias_bo,
                                   pw_output_bo, B, CI, H, W, CO, true);
            batchnorm2d_fpga_forward(pw_output_bo, bn_weight_bo, bn_bias_bo,
                                     bn_mean_bo, bn_var_bo, bn_output_bo,
                                     B, CO, H, W, EPS, true, true);
            relu6_fpga_forward(bn_output_bo, output_bo,
                               static_cast<int>(OUTPUT_COUNT));

            // Unicamente la salida final vuelve al host para validacion.
            output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const auto* output = output_bo.map<float*>();
            double sum_sq = 0.0, max_error = 0.0;
            for (std::size_t i = 0; i < OUTPUT_COUNT; ++i) {
                const double error = std::abs(static_cast<double>(output[i]) - reference[i]);
                sum_sq += error * error;
                max_error = std::max(max_error, error);
            }
            const double rmse = std::sqrt(sum_sq / OUTPUT_COUNT);
            std::cout << "[Pointwise -> BatchNorm2D -> ReLU6] RMSE: " << rmse
                      << " | error maximo: " << max_error
                      << " | " << (rmse <= 1e-3 ? "PASS" : "FAIL") << '\n';
            if (rmse > 1e-3) return 1;
        }

        // ============================================================
        // PRUEBA RESIDUAL: ENTRADA -> POINTWISE -> LAYERADD -> RELU6
        // ============================================================
        // input_bo alimenta a Pointwise y permanece disponible hasta LayerAdd.
        // No hay lectura a CPU de resultados intermedios.
        {
            constexpr int B = 1, C = 2, H = 2, W = 3;
            constexpr int SIZE = B * C * H * W;
            constexpr float ALPHA = 0.75f;
            constexpr std::size_t WEIGHTS = C * C;
            const std::size_t data_bytes = SIZE * sizeof(float);

            xrt::kernel layeradd(device, uuid, "layer_add_forward");
            xrt::bo input_bo(device, data_bytes, pointwise.group_id(0));
            xrt::bo weight_bo(device, WEIGHTS * sizeof(float), pointwise.group_id(1));
            xrt::bo bias_bo(device, C * sizeof(float), pointwise.group_id(2));
            xrt::bo pointwise_output_bo(device, data_bytes, pointwise.group_id(3));
            xrt::bo add_output_bo(device, data_bytes, layeradd.group_id(2));
            xrt::bo final_bo(device, data_bytes, relu6.group_id(1));

            float* input = input_bo.map<float*>();
            float* weights = weight_bo.map<float*>();
            float* bias = bias_bo.map<float*>();
            for (int i = 0; i < SIZE; ++i)
                input[i] = static_cast<float>(i - 5) * 0.2f;
            // W[oc,ic]: transformacion 1x1 sin cambiar numero de canales.
            weights[0] = 0.4f; weights[1] = -0.2f;
            weights[2] = 0.1f; weights[3] = 0.6f;
            bias[0] = 0.05f; bias[1] = -0.1f;

            std::vector<float> reference(SIZE);
            for (int oc = 0; oc < C; ++oc)
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x) {
                        const int index = (oc * H + y) * W + x;
                        float pw = bias[oc];
                        for (int ic = 0; ic < C; ++ic)
                            pw += input[(ic * H + y) * W + x] * weights[oc * C + ic];
                        // LayerAdd(input, pointwise_output, ALPHA)
                        const float added = input[index] + ALPHA * pw;
                        reference[index] = std::min(6.0f, std::max(0.0f, added));
                    }

            input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bias_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            pointwise_fpga_forward(input_bo, weight_bo, bias_bo,
                                   pointwise_output_bo, B, C, H, W, C, true);
            // input_bo continua vivo: es la rama residual.
            layeradd_fpga_forward(input_bo, pointwise_output_bo,
                                  add_output_bo, SIZE, ALPHA);
            relu6_fpga_forward(add_output_bo, final_bo, SIZE);

            // Solo al final se recupera el resultado.
            final_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const float* result = final_bo.map<float*>();
            double sum_sq = 0.0, max_error = 0.0;
            for (int i = 0; i < SIZE; ++i) {
                const double error = std::abs(static_cast<double>(result[i]) - reference[i]);
                sum_sq += error * error;
                max_error = std::max(max_error, error);
            }
            const double rmse = std::sqrt(sum_sq / SIZE);
            std::cout << "[Pointwise -> LayerAdd (residual) -> ReLU6] RMSE: "
                      << rmse << " | error maximo: " << max_error
                      << " | " << (rmse <= 1e-3 ? "PASS" : "FAIL") << '\n';
            if (rmse > 1e-3) return 1;
        }

        // ============================================================
        // PRUEBA DEL RUNTIME: BUFFERS IDENTIFICADOS Y ENCADENADOS
        // ============================================================
        {
            constexpr int N = 16;
            constexpr std::size_t BYTES = N * sizeof(float);

            auto& a = runtime.create_buffer("runtime_test_input", BYTES,
                                             relu6.group_id(0));
            runtime.create_buffer("runtime_test_intermedio", BYTES,
                                  relu6.group_id(1));
            runtime.create_buffer("runtime_test_salida", BYTES,
                                  tanh.group_id(1));

            float* values = a.map<float*>();
            std::vector<float> expected(N);
            for (int i = 0; i < N; ++i) {
                values[i] = static_cast<float>(i) - 8.0f;
                expected[i] = std::tanh(std::min(6.0f,
                                          std::max(0.0f, values[i])));
            }
            a.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            relu6_fpga_forward(runtime.buffer("runtime_test_input"),
                               runtime.buffer("runtime_test_intermedio"), N);
            tanh_fpga_forward(runtime.buffer("runtime_test_intermedio"),
                              runtime.buffer("runtime_test_salida"), N);

            // Única recuperación: al finalizar ambas operaciones.
            auto& result_bo = runtime.buffer("runtime_test_salida");
            result_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const float* actual = result_bo.map<float*>();
            double sum_sq = 0.0;
            for (int i = 0; i < N; ++i) {
                const double e = static_cast<double>(actual[i]) - expected[i];
                sum_sq += e * e;
            }
            const double rmse = std::sqrt(sum_sq / N);
            std::cout << "[Runtime: ReLU6 -> Tanh] RMSE: " << rmse
                      << " | " << (rmse <= 1e-3 ? "PASS" : "FAIL") << '\n';
            if (rmse > 1e-3) return 1;

            // Liberación explícita, una vez que no hay consumidores pendientes.
            runtime.release_buffer("runtime_test_salida");
            runtime.release_buffer("runtime_test_intermedio");
            runtime.release_buffer("runtime_test_input");
            if (runtime.buffer_count() != 0) return 1;
        }


        // ============================================================
        // PRUEBA DEL CICLO DE VIDA: RAMA RESIDUAL Y REUTILIZACIÓN
        // ============================================================
        {
            constexpr int N = 16;
            constexpr std::size_t BYTES = N * sizeof(float);
            const int GROUP = relu6.group_id(0);
            const std::size_t before_reuse = runtime.reuse_count();

            // A es leído por dos operaciones distintas: no puede reciclarse
            // después de que termine la primera.
            auto& a = runtime.create_buffer("lifetime_A", BYTES, GROUP);
            runtime.configure_lifetime("lifetime_A", 2);
            auto& b = runtime.create_buffer("lifetime_B", BYTES,
                                            relu6.group_id(1));
            runtime.configure_lifetime("lifetime_B", 1);
            auto& c = runtime.create_buffer("lifetime_C", BYTES,
                                            tanh.group_id(1));
            runtime.configure_lifetime("lifetime_C", 1);

            auto* input = a.map<float*>();
            for (int i = 0; i < N; ++i)
                input[i] = static_cast<float>(i) - 8.0f;
            a.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            relu6_fpga_forward(runtime.buffer("lifetime_A"),
                               runtime.buffer("lifetime_B"), N);
            runtime.consume_buffer("lifetime_A"); // todavía queda un lector
            if (!runtime.has_buffer("lifetime_A"))
                throw std::runtime_error("A se liberó prematuramente");

            // Esta operación representa un segundo consumidor de A.
            // Puede ejecutarse de forma independiente de la salida de ReLU6.
            tanh_fpga_forward(runtime.buffer("lifetime_A"),
                              runtime.buffer("lifetime_C"), N);
            runtime.consume_buffer("lifetime_A"); // ahora ya puede reciclarse
            if (runtime.has_buffer("lifetime_A"))
                throw std::runtime_error("A no se recicló al terminar sus lectores");

            // B y C se siguen manteniendo hasta su último consumidor.
            // B será leído por Tanh y C se recuperará para validar.
            tanh_fpga_forward(runtime.buffer("lifetime_B"),
                              runtime.buffer("lifetime_C"), N);
            runtime.consume_buffer("lifetime_B");

            runtime.buffer("lifetime_C").sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const auto* got = runtime.buffer("lifetime_C").map<float*>();
            double sum_sq = 0.0;
            for (int i = 0; i < N; ++i) {
                const float v = static_cast<float>(i) - 8.0f;
                const double expected = std::tanh(std::min(6.0f, std::max(0.0f, v)));
                const double d = got[i] - expected;
                sum_sq += d * d;
            }
            const double rmse = std::sqrt(sum_sq / N);
            if (rmse > 1e-3) throw std::runtime_error("Error numérico de ciclo de vida");
            runtime.consume_buffer("lifetime_C");

            auto& reused = runtime.create_buffer("lifetime_reused", BYTES, GROUP);
            (void)reused;
            if (runtime.reuse_count() <= before_reuse)
                throw std::runtime_error("No se reutilizó ningún BO libre");
            runtime.release_buffer("lifetime_reused");
            if (runtime.buffer_count() != 0)
                throw std::runtime_error("Quedaron buffers activos");
            std::cout << "[Runtime: ciclo de vida y reutilización] RMSE: "
                      << rmse << " | PASS\n";
        }

        // ============================================================
        // REGISTRO DAG: DEPENDENCIAS Y RAMA RESIDUAL (SIN EJECUCIÓN)
        // ============================================================
        {
            runtime.clear_graph();
            runtime.add_graph_input("dag_A");

            OperationNode add;
            add.operation = "LayerAdd";
            add.inputs = {"dag_B", "dag_A"};
            add.output = "dag_C";
            add.float_args["alpha"] = 1.0f;

            OperationNode relu;
            relu.operation = "ReLU6";
            relu.inputs = {"dag_A"};
            relu.output = "dag_B";

            // Se registran en orden inverso para validar la ordenación.
            runtime.add_operation(add);
            runtime.add_operation(relu);
            runtime.set_graph_outputs({"dag_C"});
            runtime.finalize_graph();

            const auto& order = runtime.graph_execution_order();
            if (order.size() != 2 || order[0] != 1 || order[1] != 0 ||
                runtime.graph_consumers("dag_A") != 2 ||
                runtime.graph_consumers("dag_B") != 1 ||
                runtime.graph_consumers("dag_C") != 1)
                throw std::runtime_error("Error en dependencias del DAG");

            bool cycle_rejected = false;
            runtime.clear_graph();
            OperationNode p;
            p.operation = "ReLU6";
            p.inputs = {"cycle_Y"};
            p.output = "cycle_X";
            OperationNode q;
            q.operation = "Tanh";
            q.inputs = {"cycle_X"};
            q.output = "cycle_Y";
            runtime.add_operation(p);
            runtime.add_operation(q);
            runtime.set_graph_outputs({"cycle_Y"});
            try { runtime.finalize_graph(); }
            catch (const std::invalid_argument&) { cycle_rejected = true; }
            if (!cycle_rejected)
                throw std::runtime_error("El DAG no detectó un ciclo");
            runtime.clear_graph();
            std::cout << "[Runtime DAG: registro, consumidores y ciclos] PASS\n";
        }

        // ============================================================
        // DAG + BUFFER LIFETIME AUTOMÁTICO: RAMA RESIDUAL REAL
        // ============================================================
        {
            constexpr int N = 16;
            constexpr std::size_t BYTES = N * sizeof(float);
            constexpr float ALPHA = 0.75f;

            runtime.clear_graph();
            runtime.add_graph_input("mem_A");

            OperationNode activation;
            activation.operation = "ReLU6";
            activation.inputs = {"mem_A"};
            activation.output = "mem_B";

            OperationNode residual;
            residual.operation = "LayerAdd";
            residual.inputs = {"mem_B", "mem_A"};
            residual.output = "mem_C";
            residual.float_args["alpha"] = ALPHA;

            OperationNode final_activation;
            final_activation.operation = "Tanh";
            final_activation.inputs = {"mem_C"};
            final_activation.output = "mem_D";

            // Orden deliberadamente invertido para probar la planificación.
            runtime.add_operation(final_activation); // nodo 0
            runtime.add_operation(residual);         // nodo 1
            runtime.add_operation(activation);       // nodo 2
            runtime.set_graph_outputs({"mem_D"});
            runtime.finalize_graph();
            if (runtime.graph_consumers("mem_A") != 2)
                throw std::runtime_error("El DAG no detectó la rama residual");

            auto& input_bo = runtime.create_buffer("mem_A", BYTES, relu6.group_id(0));
            auto* data = input_bo.map<float*>();
            for (int i = 0; i < N; ++i)
                data[i] = static_cast<float>(i - 8) * 0.4f;
            input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            runtime.begin_graph_execution();

            // 1. ReLU6(A) -> B. A sigue vivo por su consumidor LayerAdd.
            runtime.create_graph_output_buffer("mem_B", BYTES, relu6.group_id(1));
            relu6_fpga_forward(runtime.buffer("mem_A"), runtime.buffer("mem_B"), N);
            runtime.complete_graph_operation(2);
            if (!runtime.has_buffer("mem_A"))
                throw std::runtime_error("Se liberó la entrada residual demasiado pronto");

            // 2. LayerAdd(B,A) -> C. A y B quedan libres después de run.wait().
            runtime.create_graph_output_buffer("mem_C", BYTES, layeradd.group_id(2));
            layeradd_fpga_forward(runtime.buffer("mem_B"), runtime.buffer("mem_A"),
                                  runtime.buffer("mem_C"), N, ALPHA);
            runtime.complete_graph_operation(1);
            if (runtime.has_buffer("mem_A") || runtime.has_buffer("mem_B"))
                throw std::runtime_error("No se reciclaron los buffers consumidos");

            const auto before = runtime.reuse_count();
            runtime.create_buffer("mem_prueba_reuso", BYTES, relu6.group_id(0));
            if (runtime.reuse_count() <= before)
                throw std::runtime_error("No se reutilizó el buffer liberado por el DAG");
            runtime.release_buffer("mem_prueba_reuso");

            // 3. Tanh(C) -> D. Solo D se devuelve a CPU.
            runtime.create_graph_output_buffer("mem_D", BYTES, tanh.group_id(1));
            tanh_fpga_forward(runtime.buffer("mem_C"), runtime.buffer("mem_D"), N);
            runtime.complete_graph_operation(0);
            runtime.end_graph_execution();

            auto& output_bo = runtime.buffer("mem_D");
            output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const auto* actual = output_bo.map<float*>();
            double sum_sq = 0.0;
            for (int i = 0; i < N; ++i) {
                const float x = static_cast<float>(i - 8) * 0.4f;
                const double ref = std::tanh(std::min(6.0f, std::max(0.0f, x))
                                              + ALPHA * x);
                const double err = actual[i] - ref;
                sum_sq += err * err;
            }
            const double rmse = std::sqrt(sum_sq / N);
            if (rmse > 1e-3)
                throw std::runtime_error("Error numérico en ejecución DAG residual");
            runtime.consume_buffer("mem_D"); // Reserva de salida liberada al recuperar.
            if (runtime.buffer_count() != 0)
                throw std::runtime_error("Quedaron buffers activos tras ejecutar el DAG");
            runtime.clear_graph();
            std::cout << "[Runtime DAG: memoria automática residual] RMSE: "
                      << rmse << " | PASS\n";
        }

        // ============================================================
        // DAG DIFERIDO: DESPACHO AUTOMÁTICO DE RAMA RESIDUAL
        // ============================================================
        {
            constexpr int N = 16;
            constexpr std::size_t BYTES = N * sizeof(float);
            constexpr float ALPHA = 0.75f;

            runtime.clear_graph();
            runtime.add_graph_input("auto_A");
            OperationNode relu;
            relu.operation = "ReLU6";
            relu.inputs = {"auto_A"};
            relu.output = "auto_B";
            relu.int_args["size"] = N;

            OperationNode add;
            add.operation = "LayerAdd";
            add.inputs = {"auto_B", "auto_A"};
            add.output = "auto_C";
            add.int_args["size"] = N;
            add.float_args["alpha"] = ALPHA;

            OperationNode tanh_node;
            tanh_node.operation = "Tanh";
            tanh_node.inputs = {"auto_C"};
            tanh_node.output = "auto_D";
            tanh_node.int_args["size"] = N;

            // Registro no topológico: el runtime calcula el orden.
            runtime.add_operation(tanh_node);
            runtime.add_operation(add);
            runtime.add_operation(relu);
            runtime.set_graph_outputs({"auto_D"});
            runtime.finalize_graph();

            auto& input = runtime.create_buffer("auto_A", BYTES, relu6.group_id(0));
            auto* data = input.map<float*>();
            for (int i = 0; i < N; ++i)
                data[i] = static_cast<float>(i - 8) * 0.4f;
            input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // Única llamada que ejecuta los 3 nodos y recicla los temporales.
            runtime.execute_graph();
            if (runtime.has_buffer("auto_A") || runtime.has_buffer("auto_B") ||
                runtime.has_buffer("auto_C") || !runtime.has_buffer("auto_D"))
                throw std::runtime_error("Ciclo de vida incorrecto tras execute_graph");

            auto& final_bo = runtime.buffer("auto_D");
            final_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const auto* actual = final_bo.map<float*>();
            double squared_error = 0.0;
            for (int i = 0; i < N; ++i) {
                const float x = static_cast<float>(i - 8) * 0.4f;
                const double ref = std::tanh(std::min(6.0f, std::max(0.0f, x))
                                              + ALPHA * x);
                const double error = actual[i] - ref;
                squared_error += error * error;
            }
            const double rmse = std::sqrt(squared_error / N);
            if (rmse > 1e-3)
                throw std::runtime_error("Resultado incorrecto del despachador DAG");
            runtime.consume_buffer("auto_D");
            runtime.clear_graph();
            if (runtime.buffer_count() != 0)
                throw std::runtime_error("Buffers activos después del DAG automático");
            std::cout << "[Runtime DAG: execute_graph residual automático] RMSE: "
                      << rmse << " | PASS\n";
        }

        // ============================================================
        // DAG: PESOS Y BIAS PERSISTENTES ENTRE DOS INFERENCIAS
        // ============================================================
        {
            constexpr int BATCH = 2;
            constexpr int IN = 4;
            constexpr int OUT = 3;
            constexpr std::size_t INPUT_COUNT = BATCH * IN;
            constexpr std::size_t WEIGHT_COUNT = OUT * IN;
            constexpr std::size_t BIAS_COUNT = OUT;
            constexpr std::size_t OUTPUT_COUNT = BATCH * OUT;
            const float weights[WEIGHT_COUNT] = {
                0.05f, 0.10f, 0.15f, 0.20f,
                0.25f, 0.30f, 0.35f, 0.40f,
                0.45f, 0.50f, 0.55f, 0.60f
            };
            const float bias[BIAS_COUNT] = {0.0f, 0.1f, 0.2f};

            runtime.clear_graph();
            runtime.add_graph_input("repeat_input");
            runtime.add_graph_parameter("repeat_weight");
            runtime.add_graph_parameter("repeat_bias");

            OperationNode node;
            node.operation = "Linear";
            node.inputs = {"repeat_input"};
            node.parameters = {"repeat_weight", "repeat_bias"};
            node.output = "repeat_output";
            node.int_args = {{"batch_size", BATCH},
                             {"in_features", IN},
                             {"out_features", OUT},
                             {"use_bias", 1}};
            runtime.add_operation(node);
            runtime.set_graph_outputs({"repeat_output"});
            runtime.finalize_graph();

            runtime.load_parameter("repeat_weight", weights, WEIGHT_COUNT,
                                   linear.group_id(1));
            runtime.load_parameter("repeat_bias", bias, BIAS_COUNT,
                                   linear.group_id(2));
            const auto weight_bo_handle = runtime.buffer("repeat_weight").address();
            const auto bias_bo_handle = runtime.buffer("repeat_bias").address();

            for (int execution = 1; execution <= 2; ++execution) {
                auto& input_bo = runtime.create_buffer("repeat_input",
                    INPUT_COUNT * sizeof(float), linear.group_id(0));
                auto* input_data = input_bo.map<float*>();
                for (std::size_t i = 0; i < INPUT_COUNT; ++i)
                    input_data[i] = static_cast<float>(i + 1) * (execution == 1 ? 0.1f : -0.2f);
                // Referencia CPU antes de que el DAG recicle repeat_input.
                std::vector<float> expected_values(OUTPUT_COUNT);
                for (int b = 0; b < BATCH; ++b)
                    for (int o = 0; o < OUT; ++o) {
                        float expected = bias[o];
                        for (int i = 0; i < IN; ++i)
                            expected += input_data[b * IN + i] * weights[o * IN + i];
                        expected_values[b * OUT + o] = expected;
                    }
                input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                runtime.execute_graph();
                if (!runtime.has_buffer("repeat_weight") ||
                    !runtime.has_buffer("repeat_bias") ||
                    runtime.has_buffer("repeat_input"))
                    throw std::runtime_error("Los parámetros no se conservaron entre ejecuciones");
                if (runtime.buffer("repeat_weight").address() != weight_bo_handle ||
                    runtime.buffer("repeat_bias").address() != bias_bo_handle)
                    throw std::runtime_error("Se reemplazó un BO persistente");

                auto& out_bo = runtime.buffer("repeat_output");
                out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                const auto* actual = out_bo.map<float*>();
                double squared_error = 0.0;
                for (int b = 0; b < BATCH; ++b)
                    for (int o = 0; o < OUT; ++o) {
                        const double error = static_cast<double>(actual[b * OUT + o])
                                           - expected_values[b * OUT + o];
                        squared_error += error * error;
                    }
                const double rmse = std::sqrt(squared_error / OUTPUT_COUNT);
                if (rmse > 1e-3)
                    throw std::runtime_error("Error numérico en Linear DAG repetido");
                runtime.consume_buffer("repeat_output");
                std::cout << "[Runtime DAG: parámetros persistentes, inferencia "
                          << execution << "] RMSE: " << rmse << " | PASS\n";
            }
            runtime.release_buffer("repeat_weight");
            runtime.release_buffer("repeat_bias");
            runtime.clear_graph();
            if (runtime.buffer_count() != 0)
                throw std::runtime_error("Quedaron buffers activos tras prueba persistente");
        }

        // ============================================================
        // DAG: RECUPERACIÓN DESPUÉS DE UN ERROR EN EL SEGUNDO NODO
        // ============================================================
        {
            constexpr int BATCH = 1, IN = 4, OUT = 3;
            constexpr std::size_t IN_BYTES = BATCH * IN * sizeof(float);
            constexpr std::size_t OUT_BYTES = BATCH * OUT * sizeof(float);
            const float weights[OUT * IN] = {
                0.05f, 0.10f, 0.15f, 0.20f,
                0.25f, 0.30f, 0.35f, 0.40f,
                0.45f, 0.50f, 0.55f, 0.60f
            };
            const float bias[OUT] = {0.0f, 0.1f, 0.2f};

            runtime.clear_graph();
            runtime.add_graph_input("error_input");
            runtime.add_graph_parameter("error_weight");
            runtime.add_graph_parameter("error_bias");

            OperationNode linear_node;
            linear_node.operation = "Linear";
            linear_node.inputs = {"error_input"};
            linear_node.parameters = {"error_weight", "error_bias"};
            linear_node.output = "error_intermediate";
            linear_node.int_args = {{"batch_size", BATCH},
                                    {"in_features", IN},
                                    {"out_features", OUT},
                                    {"use_bias", 1}};
            OperationNode tanh_node;
            tanh_node.operation = "Tanh";
            tanh_node.inputs = {"error_intermediate"};
            tanh_node.output = "error_output";
            // Falta int_args["size"] intencionalmente: se lanza
            // una excepción después de completar el nodo Linear.

            runtime.add_operation(linear_node);
            runtime.add_operation(tanh_node);
            runtime.set_graph_outputs({"error_output"});
            runtime.finalize_graph();
            runtime.load_parameter("error_weight", weights, OUT * IN,
                                   linear.group_id(1));
            runtime.load_parameter("error_bias", bias, OUT,
                                   linear.group_id(2));
            const auto weight_address = runtime.buffer("error_weight").address();
            const auto bias_address = runtime.buffer("error_bias").address();

            auto prepare_input = [&]() {
                auto& bo = runtime.create_buffer("error_input", IN_BYTES,
                                                 linear.group_id(0));
                auto* data = bo.map<float*>();
                for (int i = 0; i < IN; ++i)
                    data[i] = (i + 1) * 0.1f;
                bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            };
            prepare_input();
            bool caught = false;
            try {
                runtime.execute_graph();
            } catch (const std::invalid_argument&) {
                caught = true;
            }
            if (!caught || runtime.has_buffer("error_input") ||
                runtime.has_buffer("error_intermediate") ||
                runtime.has_buffer("error_output") ||
                !runtime.has_buffer("error_weight") ||
                !runtime.has_buffer("error_bias") ||
                runtime.buffer("error_weight").address() != weight_address ||
                runtime.buffer("error_bias").address() != bias_address)
                throw std::runtime_error("Error en recuperación del DAG");

            // Reparar el DAG sin recrear el runtime ni recargar los parámetros.
            runtime.clear_graph();
            runtime.add_graph_input("error_input");
            runtime.add_graph_parameter("error_weight");
            runtime.add_graph_parameter("error_bias");
            tanh_node.int_args["size"] = OUT;
            runtime.add_operation(linear_node);
            runtime.add_operation(tanh_node);
            runtime.set_graph_outputs({"error_output"});
            runtime.finalize_graph();
            prepare_input();
            runtime.execute_graph();
            auto& result = runtime.buffer("error_output");
            result.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const auto* actual = result.map<float*>();
            double squared_error = 0.0;
            for (int o = 0; o < OUT; ++o) {
                float reference = bias[o];
                for (int i = 0; i < IN; ++i)
                    reference += ((i + 1) * 0.1f) * weights[o * IN + i];
                const double error = actual[o] - std::tanh(reference);
                squared_error += error * error;
            }
            const double rmse = std::sqrt(squared_error / OUT);
            if (rmse > 1e-3)
                throw std::runtime_error("Error numérico tras recuperación del DAG");
            runtime.consume_buffer("error_output");
            runtime.release_buffer("error_weight");
            runtime.release_buffer("error_bias");
            runtime.clear_graph();
            if (runtime.buffer_count() != 0)
                throw std::runtime_error("Buffers activos tras recuperación del DAG");
            std::cout << "[Runtime DAG: recuperación tras error] RMSE: "
                      << rmse << " | PASS\n";
        }

        std::cout << "Todas las pruebas: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}