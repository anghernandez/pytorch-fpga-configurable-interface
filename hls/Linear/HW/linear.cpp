#include "linear.hpp"

void linear_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    bool use_bias
)
{
#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem0 depth=81920
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem1 depth=1280000
#pragma HLS INTERFACE m_axi port=bias   offset=slave bundle=gmem2 depth=1000
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem3 depth=64000

#pragma HLS INTERFACE s_axilite port=input        bundle=control
#pragma HLS INTERFACE s_axilite port=weight       bundle=control
#pragma HLS INTERFACE s_axilite port=bias         bundle=control
#pragma HLS INTERFACE s_axilite port=output       bundle=control
#pragma HLS INTERFACE s_axilite port=batch_size   bundle=control
#pragma HLS INTERFACE s_axilite port=in_features  bundle=control
#pragma HLS INTERFACE s_axilite port=out_features bundle=control
#pragma HLS INTERFACE s_axilite port=use_bias     bundle=control
#pragma HLS INTERFACE s_axilite port=return       bundle=control

    for (
        int sample_index = 0;
        sample_index < batch_size;
        sample_index++
    ) {
        for (
            int output_index = 0;
            output_index < out_features;
            output_index++
        ) {
            float accumulated_value = 0.0f;

            for (
                int input_index = 0;
                input_index < in_features;
                input_index++
            ) {
#pragma HLS PIPELINE

                const int input_position =
                    sample_index * in_features
                    + input_index;

                const int weight_position =
                    output_index * in_features
                    + input_index;

                const float input_value =
                    input[input_position];

                const float weight_value =
                    weight[weight_position];

                accumulated_value += (
                    input_value * weight_value
                );
            }

            if (use_bias) {
                accumulated_value +=
                    bias[output_index];
            }

            const int output_position =
                sample_index * out_features
                + output_index;

            output[output_position] =
                accumulated_value;
        }
    }
}