#include "pointwise_fpga.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>


static void reference_pointwise(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    bool use_bias
)
{
    for (int batch_index = 0;
         batch_index < batch_size;
         batch_index++) {

        for (int output_channel = 0;
             output_channel < output_channels;
             output_channel++) {

            for (int row = 0;
                 row < input_height;
                 row++) {

                for (int column = 0;
                     column < input_width;
                     column++) {

                    float accumulated_value = 0.0f;

                    if (use_bias) {
                        accumulated_value =
                            bias[output_channel];
                    }

                    for (int input_channel = 0;
                         input_channel < input_channels;
                         input_channel++) {

                        const std::size_t input_position =
                            (((static_cast<std::size_t>(batch_index)
                            * input_channels
                            + input_channel)
                            * input_height
                            + row)
                            * input_width)
                            + column;

                        const std::size_t weight_position =
                            static_cast<std::size_t>(output_channel)
                            * input_channels
                            + input_channel;

                        accumulated_value +=
                            input[input_position]
                            * weight[weight_position];
                    }

                    const std::size_t output_position =
                        (((static_cast<std::size_t>(batch_index)
                        * output_channels
                        + output_channel)
                        * input_height
                        + row)
                        * input_width)
                        + column;

                    output[output_position] =
                        accumulated_value;
                }
            }
        }
    }
}


static bool run_test(
    const std::string& xclbin_path,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    bool use_bias
)
{
    const std::size_t input_elements =
        static_cast<std::size_t>(batch_size)
        * input_channels
        * input_height
        * input_width;

    const std::size_t weight_elements =
        static_cast<std::size_t>(output_channels)
        * input_channels;

    const std::size_t output_elements =
        static_cast<std::size_t>(batch_size)
        * output_channels
        * input_height
        * input_width;


    std::vector<float> input(input_elements);
    std::vector<float> weight(weight_elements);
    std::vector<float> bias(output_channels);

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


    for (int i = 0;
         i < output_channels;
         i++) {

        bias[i] =
            static_cast<float>(
                (i % 9) - 4
            ) * 0.01f;
    }


    reference_pointwise(
        input.data(),
        weight.data(),
        bias.data(),
        reference_output.data(),
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_channels,
        use_bias
    );


    pointwise_fpga_forward(
        input.data(),
        weight.data(),
        bias.data(),
        fpga_output.data(),
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_channels,
        use_bias,
        xclbin_path
    );


    double squared_error_sum = 0.0;
    double max_error = 0.0;
    std::size_t error_count = 0;

    constexpr double tolerance = 1e-4;


    for (std::size_t i = 0;
         i < output_elements;
         i++) {

        const double error =
            std::abs(
                static_cast<double>(
                    reference_output[i]
                )
                - static_cast<double>(
                    fpga_output[i]
                )
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
        << " | shape=("
        << batch_size << ", "
        << input_channels << ", "
        << input_height << ", "
        << input_width << ")"
        << " -> "
        << output_channels
        << " canales"
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


    if (!pass) {

        std::cout
            << "Primeros valores:\n";

        const std::size_t values_to_show =
            (output_elements < 10)
            ? output_elements
            : 10;

        for (std::size_t i = 0;
             i < values_to_show;
             i++) {

            std::cout
                << "i=" << i
                << " | ref="
                << reference_output[i]
                << " | fpga="
                << fpga_output[i]
                << '\n';
        }
    }


    return pass;
}


struct TestCase
{
    int batch_size;
    int input_channels;
    int height;
    int width;
    int output_channels;
};


int main()
{
    const std::string xclbin_path =
        "./pointwiseconv2d.xclbin";


    const std::vector<TestCase> tests = {

        // Casos pequeños / funcionales
        {1, 16,  7,  7,  16},
        {1, 24, 14, 14,  32},

        // Formas representativas para MobileNetV2
        {1, 32,  112, 112, 16},
        {1, 16,   56,  56, 96},
        {1, 24,   28,  28, 144},
        {1, 32,   14,  14, 192},
        {1, 96,    7,   7, 576},
        {1, 320,   7,   7, 1280}
    };


    constexpr bool use_bias = true;

    bool all_passed = true;


    try {

        for (const TestCase& test : tests) {

            std::cout
                << "\nProbando PointwiseConv2d "
                << "shape=("
                << test.batch_size << ", "
                << test.input_channels << ", "
                << test.height << ", "
                << test.width << ")"
                << " -> "
                << test.output_channels
                << " canales...\n";


            const bool passed =
                run_test(
                    xclbin_path,
                    test.batch_size,
                    test.input_channels,
                    test.height,
                    test.width,
                    test.output_channels,
                    use_biasy 
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
