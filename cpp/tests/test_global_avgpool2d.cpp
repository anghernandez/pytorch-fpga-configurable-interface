#include "GlobalAvgPool2d.hpp"
#include "test_utils.hpp"

#include <iostream>
#include <vector>

int main()
{
    bool passed = true;


    // =========================================================
    // Test 1: caso básico 1x1x2x2
    // =========================================================
    {
        const int batch_size = 1;
        const int channels = 1;
        const int height = 2;
        const int width = 2;

        const std::vector<float> input = {
            1.0f, 2.0f,
            3.0f, 4.0f
        };

        // (1 + 2 + 3 + 4) / 4 = 2.5
        const std::vector<float> expected = {
            2.5f
        };

        std::vector<float> output(
            batch_size * channels,
            0.0f
        );

        global_avgpool2d_forward(
            input.data(),
            output.data(),
            batch_size,
            channels,
            height,
            width
        );

        passed = check_allclose(
            "GlobalAvgPool2d basico",
            expected,
            output
        ) && passed;
    }


    // =========================================================
    // Test 2: varios canales
    // =========================================================
    {
        const int batch_size = 1;
        const int channels = 3;
        const int height = 2;
        const int width = 2;

        const std::vector<float> input = {
            // Canal 0 -> promedio 2.5
            1.0f, 2.0f,
            3.0f, 4.0f,

            // Canal 1 -> promedio 6.5
            5.0f, 6.0f,
            7.0f, 8.0f,

            // Canal 2 -> promedio 10.5
             9.0f, 10.0f,
            11.0f, 12.0f
        };

        const std::vector<float> expected = {
            2.5f,
            6.5f,
            10.5f
        };

        std::vector<float> output(
            batch_size * channels,
            0.0f
        );

        global_avgpool2d_forward(
            input.data(),
            output.data(),
            batch_size,
            channels,
            height,
            width
        );

        passed = check_allclose(
            "GlobalAvgPool2d canales",
            expected,
            output
        ) && passed;
    }


    // =========================================================
    // Test 3: batch > 1
    // =========================================================
    {
        const int batch_size = 2;
        const int channels = 2;
        const int height = 2;
        const int width = 2;

        const std::vector<float> input = {
            // Batch 0, canal 0 -> 2.5
            1.0f, 2.0f,
            3.0f, 4.0f,

            // Batch 0, canal 1 -> 6.5
            5.0f, 6.0f,
            7.0f, 8.0f,

            // Batch 1, canal 0 -> 10.5
             9.0f, 10.0f,
            11.0f, 12.0f,

            // Batch 1, canal 1 -> 14.5
            13.0f, 14.0f,
            15.0f, 16.0f
        };

        const std::vector<float> expected = {
            2.5f,
            6.5f,
            10.5f,
            14.5f
        };

        std::vector<float> output(
            batch_size * channels,
            0.0f
        );

        global_avgpool2d_forward(
            input.data(),
            output.data(),
            batch_size,
            channels,
            height,
            width
        );

        passed = check_allclose(
            "GlobalAvgPool2d batch",
            expected,
            output
        ) && passed;
    }


    // =========================================================
    // Test 4: valores positivos y negativos
    // =========================================================
    {
        const int batch_size = 1;
        const int channels = 2;
        const int height = 2;
        const int width = 2;

        const std::vector<float> input = {
            // Promedio = 0
            -3.0f, -1.0f,
             1.0f,  3.0f,

            // Promedio = -2.5
            -1.0f, -2.0f,
            -3.0f, -4.0f
        };

        const std::vector<float> expected = {
            0.0f,
            -2.5f
        };

        std::vector<float> output(
            batch_size * channels,
            0.0f
        );

        global_avgpool2d_forward(
            input.data(),
            output.data(),
            batch_size,
            channels,
            height,
            width
        );

        passed = check_allclose(
            "GlobalAvgPool2d negativos",
            expected,
            output
        ) && passed;
    }


    // =========================================================
    // Test 5: dimensiones espaciales no cuadradas
    // H = 2, W = 3
    // =========================================================
    {
        const int batch_size = 1;
        const int channels = 1;
        const int height = 2;
        const int width = 3;

        const std::vector<float> input = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        };

        // 21 / 6 = 3.5
        const std::vector<float> expected = {
            3.5f
        };

        std::vector<float> output(
            batch_size * channels,
            0.0f
        );

        global_avgpool2d_forward(
            input.data(),
            output.data(),
            batch_size,
            channels,
            height,
            width
        );

        passed = check_allclose(
            "GlobalAvgPool2d H diferente W",
            expected,
            output
        ) && passed;
    }


    // =========================================================
    // Test 6: entrada espacial 1x1
    //
    // El promedio de un único elemento debe ser él mismo.
    // =========================================================
    {
        const int batch_size = 2;
        const int channels = 3;
        const int height = 1;
        const int width = 1;

        const std::vector<float> input = {
             1.0f,
             2.0f,
             3.0f,

            -1.0f,
            -2.0f,
            -3.0f
        };

        const std::vector<float> expected = {
             1.0f,
             2.0f,
             3.0f,
            -1.0f,
            -2.0f,
            -3.0f
        };

        std::vector<float> output(
            batch_size * channels,
            0.0f
        );

        global_avgpool2d_forward(
            input.data(),
            output.data(),
            batch_size,
            channels,
            height,
            width
        );

        passed = check_allclose(
            "GlobalAvgPool2d 1x1",
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

    std::cout << "test_global_avgpool2d: PASSED\n";

    return 0;
}