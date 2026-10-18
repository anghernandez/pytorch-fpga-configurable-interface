#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

constexpr int MAX_INPUT_SIZE  = 75264;
constexpr int MAX_WEIGHT_SIZE = 2400;
constexpr int MAX_BIAS_SIZE   = 16;
constexpr int MAX_OUTPUT_SIZE = 301056;

static void reference_conv2d(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool use_bias
)
{
    for (int batch_index = 0;
         batch_index < batch_size;
         batch_index++) {

        for (int output_channel = 0;
             output_channel < output_channels;
             output_channel++) {

            for (int output_row = 0;
                 output_row < output_height;
                 output_row++) {

                for (int output_column = 0;
                     output_column < output_width;
                     output_column++) {

                    float accumulated_value = 0.0f;

                    if (use_bias) {
                        accumulated_value = bias[output_channel];
                    }

                    for (int input_channel = 0;
                         input_channel < input_channels;
                         input_channel++) {

                        for (int kernel_row = 0;
                             kernel_row < kernel_height;
                             kernel_row++) {

                            const int input_row =
                                output_row * stride_height
                                + kernel_row
                                - padding_height;

                            for (int kernel_column = 0;
                                 kernel_column < kernel_width;
                                 kernel_column++) {

                                const int input_column =
                                    output_column * stride_width
                                    + kernel_column
                                    - padding_width;

                                if (
                                    input_row >= 0
                                    && input_row < input_height
                                    && input_column >= 0
                                    && input_column < input_width
                                ) {
                                    const int input_position =
                                        (((batch_index * input_channels
                                        + input_channel)
                                        * input_height
                                        + input_row)
                                        * input_width)
                                        + input_column;

                                    const int weight_position =
                                        (((output_channel * input_channels
                                        + input_channel)
                                        * kernel_height
                                        + kernel_row)
                                        * kernel_width)
                                        + kernel_column;

                                    accumulated_value +=
                                        input[input_position]
                                        * weight[weight_position];
                                }
                            }
                        }
                    }

                    const int output_position =
                        (((batch_index * output_channels
                        + output_channel)
                        * output_height
                        + output_row)
                        * output_width)
                        + output_column;

                    output[output_position] = accumulated_value;
                }
            }
        }
    }
}

static bool run_test(
    xrt::device& device,
    xrt::kernel& kernel,
    const char* test_name,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool use_bias
)
{
    const int output_height =
        ((input_height + 2 * padding_height - kernel_height)
        / stride_height) + 1;

    const int output_width =
        ((input_width + 2 * padding_width - kernel_width)
        / stride_width) + 1;

    const int input_size =
        batch_size
        * input_channels
        * input_height
        * input_width;

    const int weight_size =
        output_channels
        * input_channels
        * kernel_height
        * kernel_width;

    const int bias_size = output_channels;

    const int output_size =
        batch_size
        * output_channels
        * output_height
        * output_width;

    std::vector<float> input(MAX_INPUT_SIZE, 0.0f);
    std::vector<float> weight(MAX_WEIGHT_SIZE, 0.0f);
    std::vector<float> bias(MAX_BIAS_SIZE, 0.0f);
    std::vector<float> output(MAX_OUTPUT_SIZE, 0.0f);
    std::vector<float> expected(output_size, 0.0f);

    for (int index = 0; index < input_size; index++) {
        input[index] =
            static_cast<float>((index % 17) - 8) * 0.05f;
    }

    for (int index = 0; index < weight_size; index++) {
        weight[index] =
            static_cast<float>((index % 11) - 5) * 0.02f;
    }

    for (int index = 0; index < bias_size; index++) {
        bias[index] =
            static_cast<float>(index - 3) * 0.01f;
    }

    reference_conv2d(
        input.data(),
        weight.data(),
        bias.data(),
        expected.data(),
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_channels,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        use_bias
    );

    xrt::bo input_bo(
        device,
        MAX_INPUT_SIZE * sizeof(float),
        kernel.group_id(0)
    );

    xrt::bo weight_bo(
        device,
        MAX_WEIGHT_SIZE * sizeof(float),
        kernel.group_id(1)
    );

    xrt::bo bias_bo(
        device,
        MAX_BIAS_SIZE * sizeof(float),
        kernel.group_id(2)
    );

    xrt::bo output_bo(
        device,
        MAX_OUTPUT_SIZE * sizeof(float),
        kernel.group_id(3)
    );

    auto input_map = input_bo.map<float*>();
    auto weight_map = weight_bo.map<float*>();
    auto bias_map = bias_bo.map<float*>();
    auto output_map = output_bo.map<float*>();

    std::memcpy(
        input_map,
        input.data(),
        MAX_INPUT_SIZE * sizeof(float)
    );

    std::memcpy(
        weight_map,
        weight.data(),
        MAX_WEIGHT_SIZE * sizeof(float)
    );

    std::memcpy(
        bias_map,
        bias.data(),
        MAX_BIAS_SIZE * sizeof(float)
    );

    std::fill(
        output_map,
        output_map + MAX_OUTPUT_SIZE,
        0.0f
    );

    input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bias_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    output_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel(
        input_bo,
        weight_bo,
        bias_bo,
        output_bo,
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_channels,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        use_bias
    );

    run.wait();

    output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    for (int index = 0; index < output_size; index++) {
        output[index] = output_map[index];
    }

    double squared_error_sum = 0.0;
    float maximum_error = 0.0f;

    for (int index = 0; index < output_size; index++) {
        const float error =
            std::fabs(output[index] - expected[index]);

        squared_error_sum +=
            static_cast<double>(error)
            * static_cast<double>(error);

        maximum_error =
            std::max(maximum_error, error);
    }

    const double rmse =
        std::sqrt(squared_error_sum / output_size);

    const float tolerance = 1.0e-5f;
    const bool passed =
        maximum_error <= tolerance;

    std::cout << "\n" << test_name << "\n";
    std::cout << "Input size  = "
              << input_size << "\n";
    std::cout << "Weight size = "
              << weight_size << "\n";
    std::cout << "Output size = "
              << output_size << "\n";

    std::cout << std::scientific
              << std::setprecision(8);

    std::cout << "RMSE        = "
              << rmse << "\n";

    std::cout << "Max error   = "
              << maximum_error << "\n";

    std::cout << "Result      = "
              << (passed ? "PASS" : "FAIL")
              << "\n";

    return passed;
}

int main()
{
    try {
        std::cout
            << "Opening FPGA device...\n";

        xrt::device device(0);

        const std::string xclbin_path =
           //"/lib/firmware/xilinx/conv2d/conv2d.xclbin";
           //"/lib/firmware/xilinx/fpga_kernels_v1.0/fpga_kernels.xclbin";
           "/lib/firmware/xilinx/fpga_accelerators/fpga_kernels.xclbin";
           std::cout
            << "Loading xclbin...\n";

        auto uuid =
            device.load_xclbin(xclbin_path);

        std::cout
            << "Opening kernel conv2d_forward...\n";

        xrt::kernel kernel(
            device,
            uuid,
            "conv2d_forward"
        );

        bool all_tests_passed = true;

        all_tests_passed &= run_test(
            device,
            kernel,
            "Test 1 - LeNet Conv1: 1x28x28 -> 6x28x28",
            64,
            1,
            28,
            28,
            6,
            5,
            5,
            1,
            1,
            2,
            2,
            true
        );

        all_tests_passed &= run_test(
            device,
            kernel,
            "Test 2 - LeNet Conv2: 6x14x14 -> 16x10x10",
            64,
            6,
            14,
            14,
            16,
            5,
            5,
            1,
            1,
            0,
            0,
            true
        );

        std::cout
            << "\nOverall result: "
            << (all_tests_passed ? "PASS" : "FAIL")
            << "\n";

        return all_tests_passed ? 0 : 1;
    }
    catch (const std::exception& exception) {
        std::cerr
            << "ERROR: "
            << exception.what()
            << "\n";

        return 1;
    }
}


