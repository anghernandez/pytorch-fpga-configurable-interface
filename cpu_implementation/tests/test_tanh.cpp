#include "Tanh.hpp"
#include "test_utils.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main()
{
    const std::vector<float> input = {
        -100.0f, -10.0f, -2.0f, -1.0f, -0.25f,
        0.0f, 0.25f, 1.0f, 2.0f, 10.0f, 100.0f
    };
    std::vector<float> expected(input.size());
    std::vector<float> output(input.size(), 0.0f);

    for (std::size_t index = 0; index < input.size(); ++index) {
        expected[index] = std::tanh(input[index]);
    }

    tanh_forward(input.data(), output.data(), static_cast<int>(input.size()));

    if (!check_allclose("Tanh", expected, output)) {
        return 1;
    }
    std::cout << "test_tanh: PASSED\n";
    return 0;
}
