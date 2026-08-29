#include "ReLU.hpp"

void relu_forward(
    const float* input,
    float* output,
    int size
)
{
    for (int index = 0; index < size; index++) {
        if (input[index] > 0.0f) {
            output[index] = input[index];
        } else {
            output[index] = 0.0f;
        }
    }
}