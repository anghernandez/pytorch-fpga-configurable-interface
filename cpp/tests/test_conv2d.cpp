#include "Conv2d.hpp"
#include "test_utils.hpp"

#include <iostream>
#include <vector>

int main()
{
    bool passed = true;

    const std::vector<float> input_channels = {
         1,  2,  3,  4,  5,  6,  7,  8,  9,
        10, 11, 12, 13, 14, 15, 16, 17, 18
    };
    const std::vector<float> weight_channels = {
        1, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 1
    };
    const std::vector<float> bias = {0.5f, -1.0f};
    std::vector<float> output_channels(8, 0.0f);

    conv2d_forward(
        input_channels.data(), weight_channels.data(), bias.data(),
        output_channels.data(), 1, 2, 3, 3, 2, 2, 2,
        2, 2, 1, 1, 0, 0, true
    );
    const std::vector<float> expected_channels = {
        12.5f, 16.5f, 24.5f, 28.5f,
        47.0f, 51.0f, 59.0f, 63.0f
    };
    passed = check_allclose("Conv2d canales y bias",
                            expected_channels, output_channels)
          && passed;

    const std::vector<float> input_padding = {1, 2, 3, 4};
    const std::vector<float> weight_padding(9, 1.0f);
    std::vector<float> output_padding(4, 0.0f);
    conv2d_forward(
        input_padding.data(), weight_padding.data(), nullptr,
        output_padding.data(), 1, 1, 2, 2, 1, 2, 2,
        3, 3, 1, 1, 1, 1, false
    );
    const std::vector<float> expected_padding = {10, 10, 10, 10};
    passed = check_allclose("Conv2d padding",
                            expected_padding, output_padding)
          && passed;

    const std::vector<float> input_batch = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<float> weight_batch = {2};
    const std::vector<float> bias_batch = {1};
    std::vector<float> output_batch(8, 0.0f);
    conv2d_forward(
        input_batch.data(), weight_batch.data(), bias_batch.data(),
        output_batch.data(), 2, 1, 2, 2, 1, 2, 2,
        1, 1, 1, 1, 0, 0, true
    );
    const std::vector<float> expected_batch = {3, 5, 7, 9, 11, 13, 15, 17};
    passed = check_allclose("Conv2d batch",
                            expected_batch, output_batch)
          && passed;

    if (!passed) {
        return 1;
    }
    std::cout << "test_conv2d: PASSED\n";
    return 0;
}
