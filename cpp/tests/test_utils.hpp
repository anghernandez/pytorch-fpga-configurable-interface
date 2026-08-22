#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

inline bool check_allclose(
    const std::string& test_name,
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float absolute_tolerance = 1e-6f,
    float rmse_limit = 1e-6f
)
{
    if (expected.size() != actual.size() || expected.empty()) {
        std::cerr << test_name << ": FAILED (tamaños inválidos)\n";
        return false;
    }

    double squared_error_sum = 0.0;
    float max_absolute_error = 0.0f;
    std::size_t max_error_index = 0;

    for (std::size_t index = 0; index < expected.size(); ++index) {
        const float error = std::fabs(expected[index] - actual[index]);
        squared_error_sum += static_cast<double>(error) * error;
        if (error > max_absolute_error) {
            max_absolute_error = error;
            max_error_index = index;
        }
    }

    const float rmse = static_cast<float>(
        std::sqrt(squared_error_sum / static_cast<double>(expected.size()))
    );
    const bool passed = max_absolute_error <= absolute_tolerance
                     && rmse <= rmse_limit;

    std::cout << std::fixed << std::setprecision(8)
              << test_name
              << ": RMSE=" << rmse
              << ", max_error=" << max_absolute_error
              << (passed ? " -> PASSED\n" : " -> FAILED\n");

    if (!passed) {
        std::cerr << "  índice=" << max_error_index
                  << ", esperado=" << expected[max_error_index]
                  << ", obtenido=" << actual[max_error_index]
                  << '\n';
    }

    return passed;
}

#endif
