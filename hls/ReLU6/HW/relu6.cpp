#include "relu6.hpp"

void relu6_forward(
    const float* input,
    float* output,
    int size
)
{
#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem0 depth=size
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=size

#pragma HLS INTERFACE s_axilite port=input
#pragma HLS INTERFACE s_axilite port=output
#pragma HLS INTERFACE s_axilite port=size
#pragma HLS INTERFACE s_axilite port=return

    for (int index = 0;
         index < size;
         index++) {
#pragma HLS PIPELINE

        const float input_value =
            input[index];

        float output_value;

        if (input_value < 0.0f) {
            output_value = 0.0f;
        }
        else if (input_value > 6.0f) {
            output_value = 6.0f;
        }
        else {
            output_value = input_value;
        }

        output[index] = output_value;
    }
}