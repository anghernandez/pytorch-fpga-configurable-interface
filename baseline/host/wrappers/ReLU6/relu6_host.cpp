#include "relu6_fpga.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>


static float reference_relu6(float value)
{
    if (value < 0.0f) {
        return 0.0f;
    }

    if (value > 6.0f) {
        return 6.0f;
    }

    return value;
}


static bool run_test(
    const std::string& xclbin_path,
    int size
)
{
    std::vector<float> input(size);
    std::vector<float> reference_output(size);
    std::vector<float> fpga_output(size, 0.0f);


    for (int i = 0; i < size; i++) {

        // Genera valores entre aproximadamente -2 y 8.
        input[i] =
            static_cast<float>(
                (i % 101) - 20
            ) * 0.1f;

        reference_output[i] =
            reference_relu6(input[i]);
    }


    relu6_fpga_forward(
        input.data(),
        fpga_output.data(),
        size,
        xclbin_path
    );


    double squared_error_sum = 0.0;
    double max_error = 0.0;
    std::size_t error_count = 0;

    constexpr double tolerance = 1e-6;


    for (int i = 0; i < size; i++) {

        const double error =
            std::abs(
                static_cast<double>(reference_output[i])
                - static_cast<double>(fpga_output[i])
            );

        squared_error_sum += error * error;

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
            / static_cast<double>(size)
        );


    const bool pass =
        error_count == 0;


    std::cout
        << (pass ? "PASS" : "FAIL")
        << " | size="
        << size
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
        "./relu6.xclbin";


    const std::vector<int> test_sizes = {
        1,
        7,
        16,
        257,
        1024,
        150528,
        401408,
        802816
    };


    bool all_passed = true;


    try {

        for (int size : test_sizes) {

            std::cout
                << "\nProbando ReLU6 size="
                << size
                << "...\n";


            if (!run_test(
                    xclbin_path,
                    size
                )) {

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
