#include "depthwise_fpga.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>


static void reference_depthwise(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
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
    bool use_bias
)
{
    for (int batch_index = 0;
         batch_index < batch_size;
         batch_index++) {

        for (int channel = 0;
             channel < channels;
             channel++) {

            for (int output_row = 0;
                 output_row < output_height;
                 output_row++) {

                for (int output_column = 0;
                     output_column < output_width;
                     output_column++) {

                    float accumulated_value = 0.0f;

                    if (use_bias) {
                        accumulated_value =
                            bias[channel];
                    }

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
                                input_row >= 0 &&
                                input_row < input_height &&
                                input_column >= 0 &&
                                input_column < input_width
                            ) {
                                const std::size_t input_position =
                                    (((static_cast<std::size_t>(batch_index)
                                    * channels
                                    + channel)
                                    * input_height
                                    + input_row)
                                    * input_width)
                                    + input_column;

                                const std::size_t weight_position =
                                    ((static_cast<std::size_t>(channel)
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

                    const std::size_t output_position =
                        (((static_cast<std::size_t>(batch_index)
                        * channels
                        + channel)
                        * output_height
                        + output_row)
                        * output_width)
                        + output_column;

                    output[output_position] =
                        accumulated_value;
                }
            }
        }
    }
}


struct TestCase
{
    int batch_size;
    int channels;
    int input_height;
    int input_width;
    int kernel_height;
    int kernel_width;
    int stride_height;
    int stride_width;
    int padding_height;
    int padding_width;
};


static bool run_test(
    const std::string& xclbin_path,
    const TestCase& test
)
{
    const int output_height =
        (
            test.input_height
            + 2 * test.padding_height
            - test.kernel_height
        ) / test.stride_height + 1;

    const int output_width =
        (
            test.input_width
            + 2 * test.padding_width
            - test.kernel_width
        ) / test.stride_width + 1;


    const std::size_t input_elements =
        static_cast<std::size_t>(test.batch_size)
        * test.channels
        * test.input_height
        * test.input_width;

    const std::size_t weight_elements =
        static_cast<std::size_t>(test.channels)
        * test.kernel_height
        * test.kernel_width;

    const std::size_t output_elements =
        static_cast<std::size_t>(test.batch_size)
        * test.channels
        * output_height
        * output_width;


    std::vector<float> input(input_elements);
    std::vector<float> weight(weight_elements);
    std::vector<float> bias(test.channels);

    std::vector<float> reference_output(
        output_elements
    );

    std::vector<float> fpga_output(
        output_elements,
        0.0f
    );


    for (std::size_t i = 0;
         i < input_elements;
         i++) {

        input[i] =
            static_cast<float>(
                static_cast<int>(i % 31) - 15
            ) * 0.01f;
    }


    for (std::size_t i = 0;
         i < weight_elements;
         i++) {

        weight[i] =
            static_cast<float>(
                static_cast<int>(i % 17) - 8
            ) * 0.005f;
    }


    for (int channel = 0;
         channel < test.channels;
         channel++) {

        bias[channel] =
            static_cast<float>(
                (channel % 9) - 4
            ) * 0.01f;
    }


    constexpr bool use_bias = true;


    reference_depthwise(
        input.data(),
        weight.data(),
        bias.data(),
        reference_output.data(),
        test.batch_size,
        test.channels,
        test.input_height,
        test.input_width,
        output_height,
        output_width,
        test.kernel_height,
        test.kernel_width,
        test.stride_height,
        test.stride_width,
        test.padding_height,
        test.padding_width,
        use_bias
    );


    depthwise_fpga_forward(
        input.data(),
        weight.data(),
        bias.data(),
        fpga_output.data(),
        test.batch_size,
        test.channels,
        test.input_height,
        test.input_width,
        output_height,
        output_width,
        test.kernel_height,
        test.kernel_width,
        test.stride_height,
        test.stride_width,
        test.padding_height,
        test.padding_width,
        use_bias,
        xclbin_path
    );


    double squared_error_sum = 0.0;
    double max_error = 0.0;
    std::size_t error_count = 0;

    constexpr double tolerance = 1e-5;


    for (std::size_t i = 0;
         i < output_elements;
         i++) {

        const double error =
            std::abs(
                static_cast<double>(reference_output[i])
                - static_cast<double>(fpga_output[i])
            );

        squared_error_sum +=
            error * error;

        if (error > max_error) {
            max_error = error;
        }

        if (error > tolerance) {
            error_count++;
        }
    }


    const double rmse =
        std::sqrt(
            squared_error_sum
            / static_cast<double>(output_elements)
        );


    const bool pass =
        error_count == 0;


    std::cout
        << (pass ? "PASS" : "FAIL")
        << " | input=("
        << test.batch_size << ", "
        << test.channels << ", "
        << test.input_height << ", "
        << test.input_width << ")"
        << " | output=("
        << test.batch_size << ", "
        << test.channels << ", "
        << output_height << ", "
        << output_width << ")"
        << " | stride="
        << test.stride_height
        << " | elementos_salida="
        << output_elements
        << " | errores="
        << error_count
        << " | max_error="
        << std::scientific
        << max_error
        << " | RMSE="
        << rmse
        << std::defaultfloat
        << '\n';


    return pass;
}


int main()
{
    const std::string xclbin_path =
        "./depthwiseconv2d.xclbin";


    const std::vector<TestCase> tests = {

        // Pruebas funcionales pequeñas
        {1, 16,  7,   7, 3, 3, 1, 1, 1, 1},
        {1, 24, 14,  14, 3, 3, 2, 2, 1, 1},

        // Formas representativas de MobileNetV2
        {1, 32, 112, 112, 3, 3, 1, 1, 1, 1},
        {1, 96,  56,  56, 3, 3, 2, 2, 1, 1},
        {1, 144, 28,  28, 3, 3, 1, 1, 1, 1},
        {1, 192, 28,  28, 3, 3, 2, 2, 1, 1},
        {1, 576, 14,  14, 3, 3, 2, 2, 1, 1},
        {1, 960,  7,   7, 3, 3, 1, 1, 1, 1}
    };


    bool all_passed = true;


    try {

        for (const TestCase& test : tests) {

            std::cout
                << "\nProbando DepthwiseConv2d "
                << "input=("
                << test.batch_size << ", "
                << test.channels << ", "
                << test.input_height << ", "
                << test.input_width << ")"
                << " | stride="
                << test.stride_height
                << "...\n";


            const bool passed =
                run_test(
                    xclbin_path,
                    test
                );


            if (!passed) {
                all_passed = false;
            }
        }

    }
    catch (const std::exception& exception) {

        std::cerr
            << "ERROR: "
            << exception.what()
            << '\n';

        return 1;
    }


    std::cout
        << "\nResultado final: "
        << (all_passed ? "PASS" : "FAIL")
        << '\n';


    return all_passed ? 0 : 1;
}
