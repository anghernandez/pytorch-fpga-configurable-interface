#include "batchnorm2d_fpga.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>


static void reference_batchnorm2d(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    float eps,
    bool affine,
    bool use_bias
)
{
    for (int batch_index = 0;
         batch_index < batch_size;
         batch_index++) {

        for (int channel_index = 0;
             channel_index < channels;
             channel_index++) {

            const float mean =
                running_mean[channel_index];

            const float variance =
                running_var[channel_index];

            const float denominator =
                std::sqrt(variance + eps);

            float gamma = 1.0f;
            float beta = 0.0f;

            if (affine) {

                gamma =
                    weight[channel_index];

                if (use_bias) {
                    beta =
                        bias[channel_index];
                }
            }

            for (int row_index = 0;
                 row_index < input_height;
                 row_index++) {

                for (int column_index = 0;
                     column_index < input_width;
                     column_index++) {

                    const int position =
                        (((batch_index * channels
                        + channel_index)
                        * input_height
                        + row_index)
                        * input_width)
                        + column_index;

                    const float normalized_value =
                        (input[position] - mean)
                        / denominator;

                    output[position] =
                        gamma
                        * normalized_value
                        + beta;
                }
            }
        }
    }
}


static bool run_test(
    const std::string& xclbin_path,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    float eps,
    bool affine,
    bool use_bias
)
{
    const std::size_t element_count =
        static_cast<std::size_t>(batch_size)
        * static_cast<std::size_t>(channels)
        * static_cast<std::size_t>(input_height)
        * static_cast<std::size_t>(input_width);


    std::vector<float> input(element_count);

    std::vector<float> weight(channels);
    std::vector<float> bias(channels);
    std::vector<float> running_mean(channels);
    std::vector<float> running_var(channels);

    std::vector<float> reference_output(
        element_count
    );

    std::vector<float> fpga_output(
        element_count,
        0.0f
    );


    /*
     * Datos deterministas.
     *
     * running_var siempre debe ser positiva.
     */
    for (std::size_t i = 0;
         i < element_count;
         i++) {

        input[i] =
            static_cast<float>(
                static_cast<int>(i % 101) - 50
            ) * 0.02f;
    }


    for (int channel = 0;
         channel < channels;
         channel++) {

        weight[channel] =
            0.75f
            + static_cast<float>(channel % 11)
            * 0.025f;

        bias[channel] =
            static_cast<float>(
                (channel % 9) - 4
            ) * 0.02f;

        running_mean[channel] =
            static_cast<float>(
                (channel % 13) - 6
            ) * 0.03f;

        running_var[channel] =
            0.5f
            + static_cast<float>(channel % 17)
            * 0.05f;
    }


    reference_batchnorm2d(
        input.data(),
        weight.data(),
        bias.data(),
        running_mean.data(),
        running_var.data(),
        reference_output.data(),
        batch_size,
        channels,
        input_height,
        input_width,
        eps,
        affine,
        use_bias
    );


    batchnorm2d_fpga_forward(
        input.data(),
        weight.data(),
        bias.data(),
        running_mean.data(),
        running_var.data(),
        fpga_output.data(),
        batch_size,
        channels,
        input_height,
        input_width,
        eps,
        affine,
        use_bias,
        xclbin_path
    );


    double squared_error_sum = 0.0;
    double max_error = 0.0;

    std::size_t error_count = 0;

    constexpr double tolerance = 1e-5;


    for (std::size_t i = 0;
         i < element_count;
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
            / static_cast<double>(
                element_count
            )
        );


    const bool pass =
        error_count == 0;


    std::cout
        << (pass ? "PASS" : "FAIL")
        << " | shape=("
        << batch_size << ", "
        << channels << ", "
        << input_height << ", "
        << input_width << ")"
        << " | elementos="
        << element_count
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
            (element_count < 10)
            ? element_count
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
    int channels;
    int height;
    int width;
};


int main()
{
    const std::string xclbin_path =
        "./batchnorm2d.xclbin";


    /*
     * Formas representativas de activaciones
     * encontradas a lo largo de MobileNetV2.
     */
    const std::vector<TestCase> tests = {

        {64, 32,   112, 112},
        {1, 96,    56,  56},
        {1, 144,   28,  28},
        {1, 576,   14,  14},
        {1, 1280,   7,   7}
    };


    constexpr float eps = 1e-5f;

    constexpr bool affine = true;
    constexpr bool use_bias = true;


    bool all_passed = true;


    try {

        for (const TestCase& test : tests) {

            std::cout
                << "\nProbando BatchNorm2d "
                << "shape=("
                << test.batch_size << ", "
                << test.channels << ", "
                << test.height << ", "
                << test.width << ")...\n";


            const bool passed =
                run_test(
                    xclbin_path,
                    test.batch_size,
                    test.channels,
                    test.height,
                    test.width,
                    eps,
                    affine,
                    use_bias
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