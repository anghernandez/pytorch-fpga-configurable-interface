#ifndef FPGA_RUNTIME_HPP
#define FPGA_RUNTIME_HPP

#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_uuid.h>

class FpgaRuntime
{
public:
    static FpgaRuntime& instance();

    xrt::device& device();
    const xrt::uuid& uuid() const;

private:
    FpgaRuntime();

    FpgaRuntime(const FpgaRuntime&) = delete;
    FpgaRuntime& operator=(const FpgaRuntime&) = delete;

    xrt::device device_;
    xrt::uuid uuid_;
};

#endif
