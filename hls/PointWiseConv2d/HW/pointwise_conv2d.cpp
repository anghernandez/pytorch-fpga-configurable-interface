#include "pointwise_conv2d.hpp"

void pointwise_conv2d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    bool use_bias
)
{
#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem0 depth=12845056
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem1 depth=409600
#pragma HLS INTERFACE m_axi port=bias   offset=slave bundle=gmem2 depth=1280
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem3 depth=77070336

#pragma HLS INTERFACE s_axilite port=input
#pragma HLS INTERFACE s_axilite port=weight
#pragma HLS INTERFACE s_axilite port=bias
#pragma HLS INTERFACE s_axilite port=output

#pragma HLS INTERFACE s_axilite port=batch_size
#pragma HLS INTERFACE s_axilite port=input_channels
#pragma HLS INTERFACE s_axilite port=input_height
#pragma HLS INTERFACE s_axilite port=input_width
#pragma HLS INTERFACE s_axilite port=output_channels
#pragma HLS INTERFACE s_axilite port=use_bias
#pragma HLS INTERFACE s_axilite port=return

    for (int batch_index = 0;
         batch_index < batch_size;
         batch_index++) {

        for (int output_channel = 0;
             output_channel < output_channels;
             output_channel++) {

            for (int output_row = 0;
                 output_row < input_height;
                 output_row++) {

                for (int output_column = 0;
                     output_column < input_width;
                     output_column++) {

                    float accumulated_value = 0.0f;

                    if (use_bias) {
                        accumulated_value = bias[output_channel];
                    }

                    for (int input_channel = 0;
                         input_channel < input_channels;
                         input_channel++) {
#pragma HLS PIPELINE

                        const int input_position =
                            (((batch_index * input_channels
                            + input_channel)
                            * input_height
                            + output_row)
                            * input_width)
                            + output_column;

                        const int weight_position =
                            (output_channel * input_channels)
                            + input_channel;

                        accumulated_value +=
                            input[input_position]
                            * weight[weight_position];
                    }

                    const int output_position =
                        (((batch_index * output_channels
                        + output_channel)
                        * input_height
                        + output_row)
                        * input_width)
                        + output_column;

                    output[output_position] =
                        accumulated_value;
                }
            }
        }
    }
}