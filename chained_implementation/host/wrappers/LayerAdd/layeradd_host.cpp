#include "layeradd_fpga.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>


static void reference_layeradd(
    const float* input1,
    const float* input2,
    float* output,
    int size,
    float alpha
)
{
    for (int i = 0; i < size; i++) {
        output[i] =
            input1[i]
            + alpha * input2[i];
    }
}


static bool run_test(
    const std::string& xclbin_path,
    int size,
    float alpha
)
{
    std::vector<float> input1(size);
    std::vector<float> input2(size);

    std::vector<float> reference_output(size);
    std::vector<float> fpga_output(size, 0.0f);


    for (int i = 0; i < size; i++) {
        input1[i] =
            static_cast<float>(i) * 0.25f - 2.0f;

        input2[i] =
            static_cast<float>((i % 7) - 3) * 0.5f;
    }


    reference_layeradd(
        input1.data(),
        input2.data(),
        reference_output.data(),
        size,
        alpha
    );


    layeradd_fpga_forward(
        input1.data(),
        input2.data(),
        fpga_output.data(),
        size,
        alpha,
        xclbin_path
    );


    double squared_error_sum = 0.0;
    double max_error = 0.0;

    int error_count = 0;

    constexpr double tolerance = 1e-5;


    for (int i = 0; i < size; i++) {

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
            / static_cast<double>(size)
        );


    const bool pass =
        error_count == 0;


    std::cout
        << (pass ? "PASS" : "FAIL")
        << " | size=" << size
        << " | alpha=" << alpha
        << " | errores=" << error_count
        << " | max_error=" << std::scientific << max_error
        << " | RMSE=" << rmse
        << std::defaultfloat
        << '\n';


    if (!pass) {

        std::cout
            << "\nPrimeros valores:\n";

        const int values_to_show =
            (size < 10) ? size : 10;

        for (int i = 0; i < values_to_show; i++) {

            std::cout
                << "i=" << i
                << " | ref=" << reference_output[i]
                << " | fpga=" << fpga_output[i]
                << '\n';
        }

        std::cout << '\n';
    }


    return pass;
}


int main()
{
    const std::string xclbin_path =
        "./layeradd.xclbin";


    const std::vector<int> test_sizes = {
        1,
        7,
        16,
        32,
        64
    };

    const float alpha = 0.75f;


    bool all_passed = true;


    try {

        for (const int size : test_sizes) {

            std::cout
                << "Probando size="
                << size
                << "...\n";

            const bool passed =
                run_test(
                    xclbin_path,
                    size,
                    alpha
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