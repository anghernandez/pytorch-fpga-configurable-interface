#include "Linear.hpp"
#include "test_utils.hpp"

#include <iostream>
#include <vector>

int main()
{
    constexpr int batch_size = 2;
    constexpr int in_features = 3;
    constexpr int out_features = 2;

    const std::vector<float> input = {
        1.0f, 2.0f, 3.0f,
       -1.0f, 0.5f, 2.0f
    };
    const std::vector<float> weight = {
        1.0f, 0.0f, -1.0f,
        0.5f, 2.0f, 1.0f
    };
    const std::vector<float> bias = {0.5f, -1.0f};

    std::vector<float> output(batch_size * out_features, 0.0f);
    linear_forward(input.data(), weight.data(), bias.data(), output.data(),
                   batch_size, in_features, out_features, true);
    const std::vector<float> expected_with_bias = {-1.5f, 6.5f, -2.5f, 1.5f};
    bool passed = check_allclose("Linear con bias", expected_with_bias, output);

    linear_forward(input.data(), weight.data(), nullptr, output.data(),
                   batch_size, in_features, out_features, false);
    const std::vector<float> expected_without_bias = {-2.0f, 7.5f, -3.0f, 2.5f};
    passed = check_allclose("Linear sin bias", expected_without_bias, output)
          && passed;

    if (!passed) {
        return 1;
    }
    std::cout << "test_linear: PASSED\n";
    return 0;
}
