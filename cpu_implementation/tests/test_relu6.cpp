#include "ReLU6.hpp"
#include "test_utils.hpp"

#include <iostream>
#include <vector>

int main()
{
    bool passed = true;

    // =========================================================
    // Test 1: comportamiento general de ReLU6
    // =========================================================

    const std::vector<float> input_values = {
        -10.0f,
        -1.0f,
         0.0f,
         0.5f,
         3.0f,
         6.0f,
         7.0f,
        10.0f
    };

    std::vector<float> output_values(
        input_values.size(),
        0.0f
    );

    relu6_forward(
        input_values.data(),
        output_values.data(),
        static_cast<int>(input_values.size())
    );

    const std::vector<float> expected_values = {
        0.0f,
        0.0f,
        0.0f,
        0.5f,
        3.0f,
        6.0f,
        6.0f,
        6.0f
    };

    passed = check_allclose(
        "ReLU6 valores",
        expected_values,
        output_values
    ) && passed;


    // =========================================================
    // Test 2: tensor linealizado de mayor tamaño
    // =========================================================

    const std::vector<float> input_tensor = {
        -2.0f,  1.0f,  2.0f,  8.0f,
         3.0f, -4.0f,  6.0f,  5.0f,
         7.0f,  0.0f, -1.0f,  4.0f
    };

    std::vector<float> output_tensor(
        input_tensor.size(),
        0.0f
    );

    relu6_forward(
        input_tensor.data(),
        output_tensor.data(),
        static_cast<int>(input_tensor.size())
    );

    const std::vector<float> expected_tensor = {
        0.0f, 1.0f, 2.0f, 6.0f,
        3.0f, 0.0f, 6.0f, 5.0f,
        6.0f, 0.0f, 0.0f, 4.0f
    };

    passed = check_allclose(
        "ReLU6 tensor",
        expected_tensor,
        output_tensor
    ) && passed;


    // =========================================================
    // Resultado
    // =========================================================

    if (!passed) {
        return 1;
    }

    std::cout << "test_relu6: PASSED\n";

    return 0;
}