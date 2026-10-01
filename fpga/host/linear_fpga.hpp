#ifndef LINEAR_FPGA_HPP
#define LINEAR_FPGA_HPP

void linear_fpga_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    bool use_bias
);

#endif
