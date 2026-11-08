#pragma once

#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace fpga_profile {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

struct Record {
    const char* operation;

    TimePoint prepare_start;
    TimePoint to_device_start;
    TimePoint to_device_end;
    TimePoint launch_end;
    TimePoint wait_end;
    TimePoint from_device_end;
    TimePoint copy_end;
};

struct State {
    bool enabled = false;
    std::vector<Record> records;
};

// Estado compartido por los archivos del módulo en el mismo hilo.
// La inferencia actual ejecuta las capas secuencialmente.
inline State& state()
{
    static thread_local State instance;
    return instance;
}

inline double milliseconds(TimePoint start, TimePoint end)
{
    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

inline void begin(std::size_t capacity)
{
    if (capacity == 0) {
        throw std::invalid_argument(
            "La capacidad del registro debe ser mayor que cero"
        );
    }

    auto& s = state();

    s.enabled = false;
    s.records.clear();
    s.records.reserve(capacity);
    s.enabled = true;
}

inline void cancel()
{
    auto& s = state();
    s.enabled = false;
    s.records.clear();
}

inline void append(
    const char* operation,
    TimePoint prepare_start,
    TimePoint to_device_start,
    TimePoint to_device_end,
    TimePoint launch_end,
    TimePoint wait_end,
    TimePoint from_device_end,
    TimePoint copy_end
)
{
    auto& s = state();

    if (!s.enabled) {
        return;
    }

    // Evita ampliar el vector durante la medición.
    if (s.records.size() == s.records.capacity()) {
        s.enabled = false;
        throw std::runtime_error(
            "Capacidad insuficiente para registrar las operaciones"
        );
    }

    s.records.push_back({
        operation,
        prepare_start,
        to_device_start,
        to_device_end,
        launch_end,
        wait_end,
        from_device_end,
        copy_end
    });
}

} // namespace fpga_profile