#include "ReLU.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <iomanip>

constexpr float RMSE_LIMIT = 1e-3f;
constexpr float ATOL = 1e-6f;
constexpr float RTOL = 1e-5f;


/*
 * Calcula:
 *
 * RMSE = sqrt(mean((reference - prediction)^2))
 */
float calculate_rmse(
    const std::vector<float>& reference,
    const std::vector<float>& prediction
)
{
    if (reference.size() != prediction.size()) {
        throw std::runtime_error(
            "No se puede calcular RMSE: tamaños diferentes."
        );
    }

    double squared_error_sum = 0.0;

    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double error =
            static_cast<double>(reference[i]) -
            static_cast<double>(prediction[i]);

        squared_error_sum += error * error;
    }

    const double mean_squared_error =
        squared_error_sum /
        static_cast<double>(reference.size());

    return static_cast<float>(
        std::sqrt(mean_squared_error)
    );
}


/*
 * Calcula el error absoluto máximo.
 */
float calculate_max_absolute_error(
    const std::vector<float>& reference,
    const std::vector<float>& prediction
)
{
    if (reference.size() != prediction.size()) {
        throw std::runtime_error(
            "No se puede calcular error máximo: tamaños diferentes."
        );
    }

    float max_error = 0.0f;

    for (std::size_t i = 0; i < reference.size(); ++i) {
        const float error = std::fabs(
            reference[i] - prediction[i]
        );

        if (error > max_error) {
            max_error = error;
        }
    }

    return max_error;
}


/*
 * Equivalente aproximado de torch.allclose().
 *
 * |a - b| <= atol + rtol * |b|
 */
bool allclose(
    const std::vector<float>& reference,
    const std::vector<float>& prediction
)
{
    if (reference.size() != prediction.size()) {
        return false;
    }

    for (std::size_t i = 0; i < reference.size(); ++i) {
        const float difference = std::fabs(
            reference[i] - prediction[i]
        );

        const float tolerance =
            ATOL + RTOL * std::fabs(reference[i]);

        if (difference > tolerance) {
            return false;
        }
    }

    return true;
}


/*
 * Convierte una forma multidimensional en
 * cantidad total de elementos.
 *
 * Ejemplo:
 *
 * 4 x 6 x 24 x 24 = 13824 elementos
 */
std::size_t calculate_num_elements(
    const std::vector<int>& shape
)
{
    std::size_t total = 1;

    for (int dimension : shape) {
        total *= static_cast<std::size_t>(dimension);
    }

    return total;
}


/*
 * Convierte la forma a texto.
 */
std::string shape_to_string(
    const std::vector<int>& shape
)
{
    std::string text = "(";

    for (std::size_t i = 0; i < shape.size(); ++i) {
        text += std::to_string(shape[i]);

        if (i + 1 < shape.size()) {
            text += ", ";
        }
    }

    text += ")";

    return text;
}


/*
 * Prueba principal de ReLU.
 */
bool test_relu(
    const std::vector<int>& shape
)
{
    const std::size_t size =
        calculate_num_elements(shape);

    /*
     * Generador reproducible.
     *
     * Equivalente conceptual a:
     *
     * torch.manual_seed(5)
     */
    std::mt19937 generator(5);

    std::normal_distribution<float> distribution(
        0.0f,
        1.0f
    );

    /*
     * Entrada compartida.
     */
    std::vector<float> input(size);

    for (float& value : input) {
        value = distribution(generator);
    }

    /*
     * Referencia.
     *
     * Este bloque representa el comportamiento
     * esperado de ReLU.
     */
    std::vector<float> reference(size);

    for (std::size_t i = 0; i < size; ++i) {
        reference[i] =
            input[i] > 0.0f
                ? input[i]
                : 0.0f;
    }

    /*
     * Ejecutar nuestra implementación C++.
     */
    std::vector<float> output(size);

    relu_forward(
        input.data(),
        output.data(),
        static_cast<int>(size)
    );

    /*
     * Calcular métricas.
     */
    const float rmse = calculate_rmse(
        reference,
        output
    );

    const float max_error =
        calculate_max_absolute_error(
            reference,
            output
        );

    const bool values_match = allclose(
        reference,
        output
    );

    const bool rmse_is_valid =
        rmse <= RMSE_LIMIT;

    /*
     * Mostrar resultados.
     */
    std::cout
        << "============================================================\n";

    std::cout
        << "VALIDACION DE ReLU C++\n";

    std::cout
        << "============================================================\n";

    std::cout
        << "Forma de entrada:       "
        << shape_to_string(shape)
        << '\n';

    std::cout
        << "Cantidad de elementos:  "
        << size
        << '\n';

    std::cout
        << std::scientific
        << std::setprecision(10);

    std::cout
        << "Error absoluto maximo:  "
        << max_error
        << '\n';

    std::cout
        << "RMSE:                   "
        << rmse
        << '\n';

    std::cout
        << "Limite RMSE:            "
        << RMSE_LIMIT
        << '\n';

    std::cout
        << std::boolalpha;

    std::cout
        << "allclose:               "
        << values_match
        << '\n';

    std::cout
        << "RMSE valido:            "
        << rmse_is_valid
        << '\n';

    const bool passed =
        values_match &&
        rmse_is_valid;

    std::cout
        << "Resultado:               "
        << (passed ? "APROBADO" : "NO APROBADO")
        << '\n';

    std::cout
        << "============================================================\n\n";

    return passed;
}


/*
 * Prueba con valores conocidos.
 */
bool test_known_values()
{
    const std::vector<float> input = {
        -3.0f,
        -1.0f,
        0.0f,
        1.0f,
        2.5f,
        -4.5f
    };

    const std::vector<float> expected = {
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        2.5f,
        0.0f
    };

    std::vector<float> output(
        input.size()
    );

    relu_forward(
        input.data(),
        output.data(),
        static_cast<int>(input.size())
    );

    return allclose(
        expected,
        output
    );
}


int main()
{
    const std::vector<std::vector<int>> test_shapes = {

        // Vector simple
        {10},

        // Salida de Linear
        {8, 16},

        // Batch similar a MNIST
        {4, 1, 28, 28},

        // Activacion similar a LeNet-5
        {4, 6, 24, 24},

        // Imagen ImageNet
        {1, 3, 224, 224},

        // Activacion similar a MobileNetV2
        {1, 32, 112, 112}
    };

    std::cout << '\n';

    std::cout
        << "############################################################\n";

    std::cout
        << "BANCO DE PRUEBAS DE ReLU C++\n";

    std::cout
        << "############################################################\n\n";


    bool all_tests_passed = true;

    int test_number = 1;

    for (const auto& shape : test_shapes) {

        const bool passed =
            test_relu(shape);

        if (!passed) {
            all_tests_passed = false;
        }

        std::cout
            << "Prueba "
            << test_number
            << ": shape="
            << shape_to_string(shape)
            << " -> "
            << (passed ? "APROBADA" : "NO APROBADA")
            << '\n';

        ++test_number;
    }


    /*
     * Valores conocidos.
     */
    const bool known_values_passed =
        test_known_values();

    if (!known_values_passed) {
        all_tests_passed = false;
    }


    std::cout << '\n';

    std::cout
        << "############################################################\n";

    std::cout
        << "RESUMEN GENERAL\n";

    std::cout
        << "############################################################\n";

    std::cout
        << "Prueba de valores conocidos: "
        << (
            known_values_passed
            ? "APROBADA"
            : "NO APROBADA"
        )
        << '\n';


    if (all_tests_passed) {

        std::cout << '\n';

        std::cout
            << "TODAS LAS PRUEBAS DE ReLU C++ "
            << "FUERON SUPERADAS.\n";

        std::cout
            << "############################################################\n";

        return 0;
    }


    std::cout << '\n';

    std::cout
        << "EXISTEN PRUEBAS NO SUPERADAS.\n";

    std::cout
        << "############################################################\n";

    return 1;
}