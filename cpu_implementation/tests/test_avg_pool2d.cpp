#include "AvgPool2d.hpp"
#include "test_utils.hpp"

#include <iostream>
#include <vector>

int main()
{
    bool passed = true;

    const std::vector<float> input_two_channels = {
         1,  2,  3,  4,  5,  6,  7,  8,
         9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32
    };
    std::vector<float> output_no_padding(8, 0.0f);
    avg_pool2d_forward(
        input_two_channels.data(), output_no_padding.data(),
        1, 2, 4, 4, 2, 2, 2, 2, 2, 2, 0, 0,
        true, false, 0
    );
    const std::vector<float> expected_no_padding = {
         3.5f,  5.5f, 11.5f, 13.5f,
        19.5f, 21.5f, 27.5f, 29.5f
    };
    passed = check_allclose("AvgPool2d sin padding",
                            expected_no_padding, output_no_padding)
          && passed;

    const std::vector<float> input_padding = {1, 2, 3, 4};
    std::vector<float> output_padding(9, 0.0f);
    avg_pool2d_forward(
        input_padding.data(), output_padding.data(),
        1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 1, 1,
        true, false, 0
    );
    const std::vector<float> expected_include_padding = {
        0.25f, 0.75f, 0.50f,
        1.00f, 2.50f, 1.50f,
        0.75f, 1.75f, 1.00f
    };
    passed = check_allclose("AvgPool2d incluyendo padding",
                            expected_include_padding, output_padding)
          && passed;

    avg_pool2d_forward(
        input_padding.data(), output_padding.data(),
        1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 1, 1,
        false, false, 0
    );
    const std::vector<float> expected_exclude_padding = {
        1.0f, 1.5f, 2.0f,
        2.0f, 2.5f, 3.0f,
        3.0f, 3.5f, 4.0f
    };
    passed = check_allclose("AvgPool2d excluyendo padding",
                            expected_exclude_padding, output_padding)
          && passed;

    if (!passed) {
        return 1;
    }
    std::cout << "test_avg_pool2d: PASSED\n";
    return 0;
}
