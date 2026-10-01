#include "fpga_runtime.hpp"

#include <string>
#include <iostream>

FpgaRuntime& FpgaRuntime::instance()
{
    static FpgaRuntime instance;
    return instance;
}

FpgaRuntime::FpgaRuntime()
    : device_(0)
{
    //const std::string xclbin_path =
        //"/home/ang/TFG-anghernandez/fpga_kernels.xclbin";

    //const std::string xclbin_path =
    //"/lib/firmware/xilinx/fpga_kernels/fpga_kernels.xclbin";

    const std::string xclbin_path =
        "/lib/firmware/xilinx/fpga_accelerators/fpga_kernels.xclbin";  

    std::cout
        << "[FpgaRuntime] Cargando: "
        << xclbin_path
        << std::endl;

    uuid_ = device_.load_xclbin(xclbin_path);
}

xrt::device& FpgaRuntime::device()
{
    return device_;
}

const xrt::uuid& FpgaRuntime::uuid() const
{
    return uuid_;
}
