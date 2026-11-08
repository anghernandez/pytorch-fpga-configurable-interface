#include "Tanh.hpp"

#include <cmath>

static float tanh_scalar(float value)
{
    if (value >= 0.0f) {
        const float exponential = std::exp(
            -2.0f * value
        );

        return (
            (1.0f - exponential)
            / (1.0f + exponential)
        );
    }

    const float exponential = std::exp(
        2.0f * value
    );

    return (
        (exponential - 1.0f)
        / (exponential + 1.0f)
    );
}

void tanh_forward(
    const float* input,
    float* output,
    int size
)
{
    for (int index = 0; index < size; index++) {
        output[index] = tanh_scalar(
            input[index]
        );
    }
}