#include "Depthwise_Conv2d.hpp"
#include "test_utils.hpp"

#include <iostream>
#include <vector>

int main()
{
    bool passed = true;

    // =========================================================
    // Test 1: canales independientes
    // =========================================================

    const std::vector<float> input_channels = {
        // Canal 0
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,

        // Canal 1
        10, 11, 12,
        13, 14, 15,
        16, 17, 18
    };

    const std::vector<float> weight_channels = {
        // Kernel canal 0
        1, 1,
        1, 1,

        // Kernel canal 1
        2, 2,
        2, 2
    };

    const std::vector<float> bias_channels = {
        0.5f,
        -1.0f
    };

    std::vector<float> output_channels(8, 0.0f);

    depthwise_conv2d_forward(
        input_channels.data(),
        weight_channels.data(),
        bias_channels.data(),
        output_channels.data(),
        1,      // batch_size
        2,      // channels
        3,      // input_height
        3,      // input_width
        2,      // output_height
        2,      // output_width
        2,      // kernel_height
        2,      // kernel_width
        1,      // stride_height
        1,      // stride_width
        0,      // padding_height
        0,      // padding_width
        true
    );

    const std::vector<float> expected_channels = {
        12.5f, 16.5f,
        24.5f, 28.5f,

        95.0f, 103.0f,
        119.0f, 127.0f
    };

    passed = check_allclose(
        "Depthwise canales y bias",
        expected_channels,
        output_channels
    ) && passed;


    // =========================================================
    // Test 2: stride = 2
    // =========================================================

    const std::vector<float> input_stride = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };

    const std::vector<float> weight_stride = {
        1, 1,
        1, 1
    };

    std::vector<float> output_stride(4, 0.0f);

    depthwise_conv2d_forward(
        input_stride.data(),
        weight_stride.data(),
        nullptr,
        output_stride.data(),
        1,      // batch_size
        1,      // channels
        4,      // input_height
        4,      // input_width
        2,      // output_height
        2,      // output_width
        2,      // kernel_height
        2,      // kernel_width
        2,      // stride_height
        2,      // stride_width
        0,      // padding_height
        0,      // padding_width
        false
    );

    const std::vector<float> expected_stride = {
        14, 22,
        46, 54
    };

    passed = check_allclose(
        "Depthwise stride",
        expected_stride,
        output_stride
    ) && passed;


    // =========================================================
    // Test 3: padding
    // =========================================================

    const std::vector<float> input_padding = {
        1, 2,
        3, 4
    };

    const std::vector<float> weight_padding(
        9,
        1.0f
    );

    std::vector<float> output_padding(4, 0.0f);

    depthwise_conv2d_forward(
        input_padding.data(),
        weight_padding.data(),
        nullptr,
        output_padding.data(),
        1,      // batch_size
        1,      // channels
        2,      // input_height
        2,      // input_width
        2,      // output_height
        2,      // output_width
        3,      // kernel_height
        3,      // kernel_width
        1,      // stride_height
        1,      // stride_width
        1,      // padding_height
        1,      // padding_width
        false
    );

    const std::vector<float> expected_padding = {
        10, 10,
        10, 10
    };

    passed = check_allclose(
        "Depthwise padding",
        expected_padding,
        output_padding
    ) && passed;


    // =========================================================
    // Test 4: batch
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

    depthwise_conv2d_forward(
        input_batch.data(),
        weight_batch.data(),
        bias_batch.data(),
        output_batch.data(),
        2,      // batch_size
        1,      // channels
        2,      // input_height
        2,      // input_width
        2,      // output_height
        2,      // output_width
        1,      // kernel_height
        1,      // kernel_width
        1,      // stride_height
        1,      // stride_width
        0,      // padding_height
        0,      // padding_width
        true
    );

    const std::vector<float> expected_batch = {
         3,  5,
         7,  9,

        11, 13,
        15, 17
    };

    passed = check_allclose(
        "Depthwise batch",
        expected_batch,
        output_batch
    ) && passed;


    // =========================================================
    // Resultado
    // =========================================================

    if (!passed) {
        return 1;
    }

    std::cout << "test_depthwise_conv2d: PASSED\n";

    return 0;
}
