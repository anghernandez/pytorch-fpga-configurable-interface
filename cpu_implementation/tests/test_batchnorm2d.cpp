#include "BatchNorm2d.hpp"
#include "test_utils.hpp"

#include <cmath>
#include <iostream>
#include <vector>


// ============================================================
// Referencia independiente para calcular el valor esperado
// ============================================================

float batchnorm_reference(
    float input,
    float gamma,
    float beta,
    float mean,
    float variance,
    float eps
)
{
    return gamma
        * ((input - mean) / std::sqrt(variance + eps))
        + beta;
}


int main()
{
    bool passed = true;


    // =========================================================
    // Test 1:
    // Un canal, affine=true, bias=true
    // =========================================================

    {
        const int batch_size = 1;
        const int channels = 1;
        const int height = 2;
        const int width = 2;

        const float eps = 1e-5f;

        const std::vector<float> input = {
            1.0f, 2.0f,
            3.0f, 4.0f
        };

        const std::vector<float> weight = {
            2.0f
        };

        const std::vector<float> bias = {
            0.5f
        };

        const std::vector<float> running_mean = {
            1.0f
        };

        const std::vector<float> running_var = {
            4.0f
        };

        std::vector<float> output(
            input.size(),
            0.0f
        );

        std::vector<float> expected(
            input.size(),
            0.0f
        );

        for (std::size_t i = 0; i < input.size(); i++) {
            expected[i] = batchnorm_reference(
                input[i],
                weight[0],
                bias[0],
                running_mean[0],
                running_var[0],
                eps
            );
        }

        batchnorm2d_forward(
            input.data(),
            weight.data(),
            bias.data(),
            running_mean.data(),
            running_var.data(),
            output.data(),
            batch_size,
            channels,
            height,
            width,
            eps,
            true,
            true
        );

        passed = check_allclose(
            "BatchNorm2d basico",
            expected,
            output
        ) && passed;
    }


    // =========================================================
    // Test 2:
    // Varios canales
    //
    // Comprueba que cada canal usa sus propios:
    // gamma, beta, mean y variance.
    // =========================================================

    {
        const int batch_size = 1;
        const int channels = 3;
        const int height = 2;
        const int width = 2;

        const float eps = 1e-5f;

        const std::vector<float> input = {
            // Canal 0
             1.0f,  2.0f,
             3.0f,  4.0f,

            // Canal 1
             5.0f,  6.0f,
             7.0f,  8.0f,

            // Canal 2
             9.0f, 10.0f,
            11.0f, 12.0f
        };

        const std::vector<float> weight = {
            1.0f,
            2.0f,
            0.5f
        };

        const std::vector<float> bias = {
             0.0f,
             1.0f,
            -2.0f
        };

        const std::vector<float> running_mean = {
            1.0f,
            5.0f,
            10.0f
        };

        const std::vector<float> running_var = {
            1.0f,
            4.0f,
            9.0f
        };

        std::vector<float> output(
            input.size(),
            0.0f
        );

        std::vector<float> expected(
            input.size(),
            0.0f
        );

        const int spatial_size =
            height * width;

        for (
            int channel = 0;
            channel < channels;
            channel++
        ) {
            for (
                int position = 0;
                position < spatial_size;
                position++
            ) {
                const int index =
                    channel * spatial_size
                    + position;

                expected[index] =
                    batchnorm_reference(
                        input[index],
                        weight[channel],
                        bias[channel],
                        running_mean[channel],
                        running_var[channel],
                        eps
                    );
            }
        }

        batchnorm2d_forward(
            input.data(),
            weight.data(),
            bias.data(),
            running_mean.data(),
            running_var.data(),
            output.data(),
            batch_size,
            channels,
            height,
            width,
            eps,
            true,
            true
        );

        passed = check_allclose(
            "BatchNorm2d varios canales",
            expected,
            output
        ) && passed;
    }


    // =========================================================
    // Test 3:
    // Batch > 1
    // =========================================================

    {
        const int batch_size = 2;
        const int channels = 1;
        const int height = 2;
        const int width = 2;

        const float eps = 1e-5f;

        const std::vector<float> input = {
            // Batch 0
            1.0f, 2.0f,
            3.0f, 4.0f,

            // Batch 1
            5.0f, 6.0f,
            7.0f, 8.0f
        };

        const std::vector<float> weight = {
            1.5f
        };

        const std::vector<float> bias = {
            0.25f
        };

        const std::vector<float> running_mean = {
            2.0f
        };

        const std::vector<float> running_var = {
            2.0f
        };

        std::vector<float> output(
            input.size(),
            0.0f
        );

        std::vector<float> expected(
            input.size(),
            0.0f
        );

        for (std::size_t i = 0; i < input.size(); i++) {
            expected[i] =
                batchnorm_reference(
                    input[i],
                    weight[0],
                    bias[0],
                    running_mean[0],
                    running_var[0],
                    eps
                );
        }

        batchnorm2d_forward(
            input.data(),
            weight.data(),
            bias.data(),
            running_mean.data(),
            running_var.data(),
            output.data(),
            batch_size,
            channels,
            height,
            width,
            eps,
            true,
            true
        );

        passed = check_allclose(
            "BatchNorm2d batch",
            expected,
            output
        ) && passed;
    }


    // =========================================================
    // Test 4:
    // affine=false
    //
    // gamma = 1
    // beta  = 0
    // =========================================================

    {
        const int batch_size = 1;
        const int channels = 2;
        const int height = 2;
        const int width = 2;

        const float eps = 1e-5f;

        const std::vector<float> input = {
            1.0f, 2.0f,
            3.0f, 4.0f,

            5.0f, 6.0f,
            7.0f, 8.0f
        };

        const std::vector<float> running_mean = {
            1.0f,
            5.0f
        };

        const std::vector<float> running_var = {
            1.0f,
            4.0f
        };

        std::vector<float> output(
            input.size(),
            0.0f
        );

        std::vector<float> expected(
            input.size(),
            0.0f
        );

        const int spatial_size =
            height * width;

        for (
            int channel = 0;
            channel < channels;
            channel++
        ) {
            for (
                int position = 0;
                position < spatial_size;
                position++
            ) {
                const int index =
                    channel * spatial_size
                    + position;

                expected[index] =
                    batchnorm_reference(
                        input[index],
                        1.0f,
                        0.0f,
                        running_mean[channel],
                        running_var[channel],
                        eps
                    );
            }
        }

        batchnorm2d_forward(
            input.data(),
            nullptr,
            nullptr,
            running_mean.data(),
            running_var.data(),
            output.data(),
            batch_size,
            channels,
            height,
            width,
            eps,
            false,
            false
        );

        passed = check_allclose(
            "BatchNorm2d affine=false",
            expected,
            output
        ) && passed;
    }


    // =========================================================
    // Test 5:
    // affine=true, bias=false
    // =========================================================

    {
        const int batch_size = 1;
        const int channels = 2;
        const int height = 1;
        const int width = 3;

        const float eps = 1e-5f;

        const std::vector<float> input = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        };

        const std::vector<float> weight = {
            2.0f,
            0.5f
        };

        const std::vector<float> running_mean = {
            1.0f,
            4.0f
        };

        const std::vector<float> running_var = {
            1.0f,
            4.0f
        };

        std::vector<float> output(
            input.size(),
            0.0f
        );

        std::vector<float> expected(
            input.size(),
            0.0f
        );

        const int spatial_size =
            height * width;

        for (
            int channel = 0;
            channel < channels;
            channel++
        ) {
            for (
                int position = 0;
                position < spatial_size;
                position++
            ) {
                const int index =
                    channel * spatial_size
                    + position;

                expected[index] =
                    batchnorm_reference(
                        input[index],
                        weight[channel],
                        0.0f,
                        running_mean[channel],
                        running_var[channel],
                        eps
                    );
            }
        }

        batchnorm2d_forward(
            input.data(),
            weight.data(),
            nullptr,
            running_mean.data(),
            running_var.data(),
            output.data(),
            batch_size,
            channels,
            height,
            width,
            eps,
            true,
            false
        );

        passed = check_allclose(
            "BatchNorm2d bias=false",
            expected,
            output
        ) && passed;
    }


    // =========================================================
    // Test 6:
    // eps configurable
    // =========================================================

    {
        const int batch_size = 1;
        const int channels = 1;
        const int height = 1;
        const int width = 4;

        const float eps = 0.1f;

        const std::vector<float> input = {
            -2.0f,
             0.0f,
             2.0f,
             4.0f
        };

        const std::vector<float> weight = {
            1.25f
        };

        const std::vector<float> bias = {
            -0.5f
        };

        const std::vector<float> running_mean = {
            1.0f
        };

        const std::vector<float> running_var = {
            0.5f
        };

        std::vector<float> output(
            input.size(),
            0.0f
        );

        std::vector<float> expected(
            input.size(),
            0.0f
        );

        for (std::size_t i = 0; i < input.size(); i++) {
            expected[i] =
                batchnorm_reference(
                    input[i],
                    weight[0],
                    bias[0],
                    running_mean[0],
                    running_var[0],
                    eps
                );
        }

        batchnorm2d_forward(
            input.data(),
            weight.data(),
            bias.data(),
            running_mean.data(),
            running_var.data(),
            output.data(),
            batch_size,
            channels,
            height,
            width,
            eps,
            true,
            true
        );

        passed = check_allclose(
            "BatchNorm2d eps configurable",
            expected,
            output
        ) && passed;
    }


    // =========================================================
    // Resultado final
    // =========================================================

    if (!passed) {
        return 1;
    }

    std::cout << "test_batchnorm2d: PASSED\n";

    return 0;
}