#include "LayerAdd.hpp"
#include "test_utils.hpp"

#include <iostream>
#include <vector>

int main()
{
    bool passed = true;

    // =========================================================
    // Test 1: suma básica, alpha = 1
    // =========================================================

    {
        const std::vector<float> input1 = {
            1.0f, 2.0f, 3.0f, 4.0f
        };

        const std::vector<float> input2 = {
            5.0f, 6.0f, 7.0f, 8.0f
        };

        const std::vector<float> expected = {
            6.0f, 8.0f, 10.0f, 12.0f
        };

        std::vector<float> output(
            input1.size(),
            0.0f
        );

        layer_add_forward(
            input1.data(),
            input2.data(),
            output.data(),
            static_cast<int>(input1.size()),
            1.0f
        );

        passed = check_allclose(
            "LayerAdd alpha=1",
            expected,
            output
        ) && passed;
    }


    // =========================================================
    // Test 2: alpha diferente de 1
    // output = input1 + 0.5 * input2
    // =========================================================

    {
        const std::vector<float> input1 = {
            1.0f, 2.0f, 3.0f, 4.0f
        };

        const std::vector<float> input2 = {
            2.0f, 4.0f, 6.0f, 8.0f
        };

        const std::vector<float> expected = {
            2.0f, 4.0f, 6.0f, 8.0f
        };

        std::vector<float> output(
            input1.size(),
            0.0f
        );

        layer_add_forward(
            input1.data(),
            input2.data(),
            output.data(),
            static_cast<int>(input1.size()),
            0.5f
        );

        passed = check_allclose(
            "LayerAdd alpha=0.5",
            expected,
            output
        ) && passed;
    }


    // =========================================================
    // Test 3: alpha = 0
    // La salida debe ser exactamente input1
    // =========================================================

    {
        const std::vector<float> input1 = {
            -3.0f, 0.0f, 2.5f, 10.0f
        };

        const std::vector<float> input2 = {
            100.0f, -50.0f, 20.0f, 5.0f
        };

        const std::vector<float> expected = {
            -3.0f, 0.0f, 2.5f, 10.0f
        };

        std::vector<float> output(
            input1.size(),
            0.0f
        );

        layer_add_forward(
            input1.data(),
            input2.data(),
            output.data(),
            static_cast<int>(input1.size()),
            0.0f
        );

        passed = check_allclose(
            "LayerAdd alpha=0",
            expected,
            output
        ) && passed;
    }


    // =========================================================
    // Test 4: alpha negativo
    // output = input1 - input2
    // =========================================================

    {
        const std::vector<float> input1 = {
            10.0f, 20.0f, 30.0f, 40.0f
        };

        const std::vector<float> input2 = {
            1.0f, 2.0f, 3.0f, 4.0f
        };

        const std::vector<float> expected = {
            9.0f, 18.0f, 27.0f, 36.0f
        };

        std::vector<float> output(
            input1.size(),
            0.0f
        );

        layer_add_forward(
            input1.data(),
            input2.data(),
            output.data(),
            static_cast<int>(input1.size()),
            -1.0f
        );

        passed = check_allclose(
            "LayerAdd alpha=-1",
            expected,
            output
        ) && passed;
    }


    // =========================================================
    // Test 5: entradas negativas y positivas
    // =========================================================

    {
        const std::vector<float> input1 = {
            -5.0f, -2.0f, 0.0f, 2.0f, 5.0f
        };

        const std::vector<float> input2 = {
             1.0f, -3.0f, 4.0f, -2.0f, 5.0f
        };

        const std::vector<float> expected = {
            -4.0f, -5.0f, 4.0f, 0.0f, 10.0f
        };

        std::vector<float> output(
            input1.size(),
            0.0f
        );

        layer_add_forward(
            input1.data(),
            input2.data(),
            output.data(),
            static_cast<int>(input1.size()),
            1.0f
        );

        passed = check_allclose(
            "LayerAdd positivos y negativos",
            expected,
            output
        ) && passed;
    }


    // =========================================================
    // Test 6: valores decimales
    // =========================================================

    {
        const std::vector<float> input1 = {
            0.1f, 0.25f, -0.5f, 1.25f
        };

        const std::vector<float> input2 = {
            0.2f, 0.5f, 0.25f, -0.5f
        };

        const std::vector<float> expected = {
            0.3f, 0.75f, -0.25f, 0.75f
        };

        std::vector<float> output(
            input1.size(),
            0.0f
        );

        layer_add_forward(
            input1.data(),
            input2.data(),
            output.data(),
            static_cast<int>(input1.size()),
            1.0f
        );

        passed = check_allclose(
            "LayerAdd decimales",
            expected,
            output
        ) && passed;
    }


    // =========================================================
    // Test 7: tensor linealizado
    //
    // Simula, por ejemplo, un tensor NCHW.
    // LayerAdd no necesita conocer sus dimensiones.
    // =========================================================

    {
        const std::vector<float> input1 = {
             1.0f,  2.0f,  3.0f,  4.0f,
             5.0f,  6.0f,  7.0f,  8.0f,
             9.0f, 10.0f, 11.0f, 12.0f
        };

        const std::vector<float> input2 = {
            12.0f, 11.0f, 10.0f, 9.0f,
             8.0f,  7.0f,  6.0f, 5.0f,
             4.0f,  3.0f,  2.0f, 1.0f
        };

        const std::vector<float> expected = {
            13.0f, 13.0f, 13.0f, 13.0f,
            13.0f, 13.0f, 13.0f, 13.0f,
            13.0f, 13.0f, 13.0f, 13.0f
        };

        std::vector<float> output(
            input1.size(),
            0.0f
        );

        layer_add_forward(
            input1.data(),
            input2.data(),
            output.data(),
            static_cast<int>(input1.size()),
            1.0f
        );

        passed = check_allclose(
            "LayerAdd tensor",
            expected,
            output
        ) && passed;
    }


    // =========================================================
    // Test 8: un solo elemento
    // =========================================================

    {
        const std::vector<float> input1 = {
            3.0f
        };

        const std::vector<float> input2 = {
            4.0f
        };

        const std::vector<float> expected = {
            11.0f
        };

        std::vector<float> output(
            input1.size(),
            0.0f
        );

        layer_add_forward(
            input1.data(),
            input2.data(),
            output.data(),
            1,
            2.0f
        );

        passed = check_allclose(
            "LayerAdd un elemento",
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

    std::cout << "test_layer_add: PASSED\n";

    return 0;
}