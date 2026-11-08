#include <iostream>
#include <string>

#include <xrt/experimental/xrt_xclbin.h>

int main()
{
    const std::string path =
        "/lib/firmware/xilinx/fpga_accelerators/fpga_kernels.xclbin";

    try
    {
        xrt::xclbin xclbin(path);

        std::cout << "========================================\n";
        std::cout << "XCLBIN INFORMATION\n";
        std::cout << "========================================\n";

        std::cout << "XSA: "
                  << xclbin.get_xsa_name()
                  << "\n";

        // --------------------------------------------------------
        // Memorias declaradas en MEM_TOPOLOGY
        // --------------------------------------------------------

        std::cout << "\n========================================\n";
        std::cout << "MEMORIAS\n";
        std::cout << "========================================\n";

        auto mems = xclbin.get_mems();

        for (const auto& mem : mems)
        {
            std::cout << "\nMemory index: "
                      << mem.get_index()
                      << "\n";

            std::cout << "  tag: "
                      << mem.get_tag()
                      << "\n";

            std::cout << "  size: "
                      << mem.get_size_kb()
                      << " KB\n";

            std::cout << "  used: "
                      << (mem.get_used() ? "yes" : "no")
                      << "\n";
        }

        // --------------------------------------------------------
        // Kernels y argumentos
        // --------------------------------------------------------

        std::cout << "\n========================================\n";
        std::cout << "KERNELS Y CONNECTIVITY\n";
        std::cout << "========================================\n";

        auto kernels = xclbin.get_kernels();

        for (const auto& kernel : kernels)
        {
            std::cout << "\n----------------------------------------\n";
            std::cout << "Kernel: "
                      << kernel.get_name()
                      << "\n";

            std::cout << "Numero de argumentos: "
                      << kernel.get_num_args()
                      << "\n";

            auto args = kernel.get_args();

            for (size_t i = 0; i < args.size(); ++i)
            {
                const auto& arg = args[i];

                std::cout << "\n  Arg "
                          << i
                          << ": "
                          << arg.get_name()
                          << "\n";

                std::cout << "    port: "
                          << arg.get_port()
                          << "\n";

                std::cout << "    size: "
                          << arg.get_size()
                          << " bytes\n";

                auto arg_mems = arg.get_mems();

                std::cout << "    connected memories:";

                for (const auto& mem : arg_mems)
                {
                    std::cout << " ["
                              << mem.get_index()
                              << " : "
                              << mem.get_tag()
                              << "]";
                }

                std::cout << "\n";
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "\nERROR: "
                  << e.what()
                  << "\n";

        return 1;
    }

    return 0;
}