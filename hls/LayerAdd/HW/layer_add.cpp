#include "layer_add.hpp"

void layer_add_forward(
    const float* input1,
    const float* input2,
    float* output,
    int size,
    float alpha
)
{
#pragma HLS INTERFACE m_axi port=input1 offset=slave bundle=gmem0 depth=64
#pragma HLS INTERFACE m_axi port=input2 offset=slave bundle=gmem1 depth=64
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem2 depth=64

#pragma HLS INTERFACE s_axilite port=input1
#pragma HLS INTERFACE s_axilite port=input2
#pragma HLS INTERFACE s_axilite port=output
#pragma HLS INTERFACE s_axilite port=size
#pragma HLS INTERFACE s_axilite port=alpha
#pragma HLS INTERFACE s_axilite port=return

    for (int index = 0;
         index < size;
         index++) {
#pragma HLS PIPELINE

        const float input1_value =
            input1[index];

        const float input2_value =
            input2[index];

        output[index] =
            input1_value
            + alpha * input2_value;
    }
}