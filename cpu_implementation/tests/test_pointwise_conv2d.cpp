#include "Pointwise_Conv2d.hpp"
#include "test_utils.hpp"

#include <iostream>
#include <vector>

int main()
{
    bool passed = true;

    // =========================================================
    // Test 1: mezcla de canales + bias
    // =========================================================

    const std::vector<float> input_channels = {
        // Canal 0
        1, 2,
        3, 4,

        // Canal 1
        5, 6,
        7, 8
    };

    const std::vector<float> weight_channels = {
        // Output channel 0: [1, 2]
        1, 2,

        // Output channel 1: [3, 4]
        3, 4
    };

    const std::vector<float> bias_channels = {
        0.5f,
        -1.0f
    };

    std::vector<float> output_channels(8, 0.0f);

    pointwise_conv2d_forward(
        input_channels.data(),
        weight_channels.data(),
        bias_channels.data(),
        output_channels.data(),
        1,      // batch_size
        2,      // input_channels
        2,      // input_height
        2,      // input_width
        2,      // output_channels
        true
    );

    const std::vector<float> expected_channels = {
        11.5f, 14.5f,
        17.5f, 20.5f,

        22.0f, 29.0f,
        36.0f, 43.0f
    };

    passed = check_allclose(
        "Pointwise canales y bias",
        expected_channels,
        output_channels
    ) && passed;


    // =========================================================
    // Test 2: sin bias
    // =========================================================

    const std::vector<float> input_no_bias = {
        1, 2,
        3, 4,

        5, 6,
        7, 8
    };

    const std::vector<float> weight_no_bias = {
        1, 1
    };

    std::vector<float> output_no_bias(4, 0.0f);

    pointwise_conv2d_forward(
        input_no_bias.data(),
        weight_no_bias.data(),
        nullptr,
        output_no_bias.data(),
        1,      // batch_size
        2,      // input_channels
        2,      // input_height
        2,      // input_width
        1,      // output_channels
        false
    );

    const std::vector<float> expected_no_bias = {
        6, 8,
        10, 12
    };

    passed = check_allclose(
        "Pointwise sin bias",
        expected_no_bias,
        output_no_bias
    ) && passed;


    // =========================================================
    // Test 3: batch
    // =========================================================

    const std::vector<float> input_batch = {
        // Batch 0
        1, 2,
        3, 4,

        // Batch 1
        5, 6,
        7, 8
    };

    const std::vector<float> weight_batch = {
        2
    };

    const std::vector<float> bias_batch = {
        1
    };

    std::vector<float> output_batch(8, 0.0f);

    pointwise_conv2d_forward(
        input_batch.data(),
        weight_batch.data(),
        bias_batch.data(),
        output_batch.data(),
        2,      // batch_size
        1,      // input_channels
        2,      // input_height
        2,      // input_width
        1,      // output_channels
        true
    );

    const std::vector<float> expected_batch = {
        3, 5,
        7, 9,

        11, 13,
        15, 17
    };

    passed = check_allclose(
        "Pointwise batch",
        expected_batch,
        output_batch
    ) && passed;


    // =========================================================
    // Resultado
    // =========================================================

    if (!passed) {
        return 1;
    }

    std::cout << "test_pointwise_conv2d: PASSED\n";

    return 0;
}