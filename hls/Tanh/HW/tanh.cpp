#include "tanh.hpp"
#include <hls_math.h>

static float tanh_scalar(float value)
{
#pragma HLS INLINE

    if (value >= 0.0f) {
        const float exponential = hls::expf(
            -2.0f * value
        );

        return (
            (1.0f - exponential)
            / (1.0f + exponential)
        );
    }

    const float exponential = hls::expf(
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
#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem0 depth=size
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=size

#pragma HLS INTERFACE s_axilite port=input  bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=size   bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    for (int index = 0; index < size; index++) {
#pragma HLS PIPELINE

        output[index] = tanh_scalar(
            input[index]
        );
    }
}