#ifndef AVG_POOL_2D_HPP
#define AVG_POOL_2D_HPP

int avg_pool2d_output_dimension(
    int input_size,
    int kernel_size,
    int stride,
    int padding,
    bool ceil_mode
);

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
);

#endif