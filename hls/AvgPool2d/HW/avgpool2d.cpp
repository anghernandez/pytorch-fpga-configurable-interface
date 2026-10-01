#include "avgpool2d.hpp"

void avg_pool2d_forward(
    const float* input,
    float* output,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool count_include_pad,
    bool use_divisor_override,
    int divisor_override
)
{

#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem0 depth=4014080
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=81920

#pragma HLS INTERFACE s_axilite port=input
#pragma HLS INTERFACE s_axilite port=output

#pragma HLS INTERFACE s_axilite port=batch_size
#pragma HLS INTERFACE s_axilite port=input_channels
#pragma HLS INTERFACE s_axilite port=input_height
#pragma HLS INTERFACE s_axilite port=input_width

#pragma HLS INTERFACE s_axilite port=output_height
#pragma HLS INTERFACE s_axilite port=output_width

#pragma HLS INTERFACE s_axilite port=kernel_height
#pragma HLS INTERFACE s_axilite port=kernel_width

#pragma HLS INTERFACE s_axilite port=stride_height
#pragma HLS INTERFACE s_axilite port=stride_width

#pragma HLS INTERFACE s_axilite port=padding_height
#pragma HLS INTERFACE s_axilite port=padding_width

#pragma HLS INTERFACE s_axilite port=count_include_pad
#pragma HLS INTERFACE s_axilite port=use_divisor_override
#pragma HLS INTERFACE s_axilite port=divisor_override

#pragma HLS INTERFACE s_axilite port=return

    for (int batch_index = 0;
         batch_index < batch_size;
         batch_index++)
    {
        for (int channel_index = 0;
             channel_index < input_channels;
             channel_index++)
        {
            for (int output_row = 0;
                 output_row < output_height;
                 output_row++)
            {
                for (int output_column = 0;
                     output_column < output_width;
                     output_column++)
                {
#pragma HLS PIPELINE

                    const int window_start_row =
                        output_row * stride_height
                        - padding_height;

                    const int window_start_column =
                        output_column * stride_width
                        - padding_width;

                    int padded_end_row =
                        window_start_row + kernel_height;

                    if (padded_end_row >
                        input_height + padding_height)
                    {
                        padded_end_row =
                            input_height + padding_height;
                    }

                    int padded_end_column =
                        window_start_column + kernel_width;

                    if (padded_end_column >
                        input_width + padding_width)
                    {
                        padded_end_column =
                            input_width + padding_width;
                    }

                    int valid_start_row =
                        window_start_row;

                    if (valid_start_row < 0)
                    {
                        valid_start_row = 0;
                    }

                    int valid_start_column =
                        window_start_column;

                    if (valid_start_column < 0)
                    {
                        valid_start_column = 0;
                    }

                    int valid_end_row =
                        padded_end_row;

                    if (valid_end_row > input_height)
                    {
                        valid_end_row = input_height;
                    }

                    int valid_end_column =
                        padded_end_column;

                    if (valid_end_column > input_width)
                    {
                        valid_end_column = input_width;
                    }

                    const int pool_height =
                        padded_end_row
                        - window_start_row;

                    const int pool_width =
                        padded_end_column
                        - window_start_column;

                    const int pool_size =
                        pool_height * pool_width;

                    float accumulated_value = 0.0f;

                    for (int input_row = valid_start_row;
                         input_row < valid_end_row;
                         input_row++)
                    {
                        for (int input_column = valid_start_column;
                             input_column < valid_end_column;
                             input_column++)
                        {
                            const int input_position =
                                (
                                    (
                                        batch_index
                                        * input_channels
                                        + channel_index
                                    )
                                    * input_height
                                    + input_row
                                )
                                * input_width
                                + input_column;

                            accumulated_value +=
                                input[input_position];
                        }
                    }

                    const int valid_height =
                        valid_end_row
                        - valid_start_row;

                    const int valid_width =
                        valid_end_column
                        - valid_start_column;

                    const int valid_value_count =
                        valid_height
                        * valid_width;

                    int divisor;

                    if (use_divisor_override)
                    {
                        divisor =
                            divisor_override;
                    }
                    else if (count_include_pad)
                    {
                        divisor =
                            pool_size;
                    }
                    else
                    {
                        divisor =
                            valid_value_count;
                    }

                    const float average_value =
                        accumulated_value
                        / (float) divisor;

                    const int output_position =
                        (
                            (
                                batch_index
                                * input_channels
                                + channel_index
                            )
                            * output_height
                            + output_row
                        )
                        * output_width
                        + output_column;

                    output[output_position] =
                        average_value;
                }
            }
        }
    }
}