#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>

#include <xrt/xrt/xrt_device.h>
#include <xrt/xrt/xrt_kernel.h>
#include <xrt/xrt/xrt_bo.h>

#define EPSILON 1e-3f


static void avg_pool2d_reference(
    const float* input,
    float* output,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool count_include_pad,
    bool use_divisor_override,
    int divisor_override
)
{
    for (int batch_index = 0;
         batch_index < batch_size;
         batch_index++)
    {
        for (int channel_index = 0;
             channel_index < input_channels;
             channel_index++)
        {
            for (int output_row = 0;
                 output_row < output_height;
                 output_row++)
            {
                for (int output_column = 0;
                     output_column < output_width;
                     output_column++)
                {
                    const int window_start_row =
                        output_row * stride_height
                        - padding_height;

                    const int window_start_column =
                        output_column * stride_width
                        - padding_width;

                    int padded_end_row =
                        window_start_row + kernel_height;

                    if (padded_end_row >
                        input_height + padding_height)
                    {
                        padded_end_row =
                            input_height + padding_height;
                    }

                    int padded_end_column =
                        window_start_column + kernel_width;

                    if (padded_end_column >
                        input_width + padding_width)
                    {
                        padded_end_column =
                            input_width + padding_width;
                    }

                    int valid_start_row =
                        window_start_row;

                    if (valid_start_row < 0)
                        valid_start_row = 0;

                    int valid_start_column =
                        window_start_column;

                    if (valid_start_column < 0)
                        valid_start_column = 0;

                    int valid_end_row =
                        padded_end_row;

                    if (valid_end_row > input_height)
                        valid_end_row = input_height;

                    int valid_end_column =
                        padded_end_column;

                    if (valid_end_column > input_width)
                        valid_end_column = input_width;

                    const int pool_height =
                        padded_end_row
                        - window_start_row;

                    const int pool_width =
                        padded_end_column
                        - window_start_column;

                    const int pool_size =
                        pool_height * pool_width;

                    float accumulated_value = 0.0f;

                    for (int input_row = valid_start_row;
                         input_row < valid_end_row;
                         input_row++)
                    {
                        for (int input_column = valid_start_column;
                             input_column < valid_end_column;
                             input_column++)
                        {
                            const int input_position =
                                (
                                    (
                                        batch_index
                                        * input_channels
                                        + channel_index
                                    )
                                    * input_height
                                    + input_row
                                )
                                * input_width
                                + input_column;

                            accumulated_value +=
                                input[input_position];
                        }
                    }

                    const int valid_height =
                        valid_end_row
                        - valid_start_row;

                    const int valid_width =
                        valid_end_column
                        - valid_start_column;

                    const int valid_value_count =
                        valid_height * valid_width;

                    int divisor;

                    if (use_divisor_override)
                    {
                        divisor =
                            divisor_override;
                    }
                    else if (count_include_pad)
                    {
                        divisor =
                            pool_size;
                    }
                    else
                    {
                        divisor =
                            valid_value_count;
                    }

                    const int output_position =
                        (
                            (
                                batch_index
                                * input_channels
                                + channel_index
                            )
                            * output_height
                            + output_row
                        )
                        * output_width
                        + output_column;

                    output[output_position] =
                        accumulated_value
                        / static_cast<float>(divisor);
                }
            }
        }
    }
}


static bool run_test(
    xrt::device& device,
    xrt::kernel& kernel,
    const char* test_name,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool count_include_pad,
    bool use_divisor_override,
    int divisor_override
)
{
    const int input_size =
        batch_size
        * input_channels
        * input_height
        * input_width;

    const int output_size =
        batch_size
        * input_channels
        * output_height
        * output_width;

    const size_t input_bytes =
        input_size * sizeof(float);

    const size_t output_bytes =
        output_size * sizeof(float);


    std::vector<float> input(input_size);
    std::vector<float> output_ref(output_size);


    for (int i = 0; i < input_size; i++)
    {
        input[i] =
            static_cast<float>(
                (i % 31) - 15
            ) / 10.0f;
    }

    for (int i = 0; i < output_size; i++)
    {
        output_ref[i] = 0.0f;
    }


    avg_pool2d_reference(
        input.data(),
        output_ref.data(),
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        count_include_pad,
        use_divisor_override,
        divisor_override
    );


    xrt::bo input_bo(
        device,
        input_bytes,
        kernel.group_id(0)
    );

    xrt::bo output_bo(
        device,
        output_bytes,
        kernel.group_id(1)
    );


    float* input_map =
        input_bo.map<float*>();

    float* output_map =
        output_bo.map<float*>();


    for (int i = 0; i < input_size; i++)
    {
        input_map[i] =
            input[i];
    }

    for (int i = 0; i < output_size; i++)
    {
        output_map[i] =
            0.0f;
    }


    input_bo.sync(
        XCL_BO_SYNC_BO_TO_DEVICE
    );


    auto run = kernel(
        input_bo,
        output_bo,
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        count_include_pad,
        use_divisor_override,
        divisor_override
    );

    run.wait();


    output_bo.sync(
        XCL_BO_SYNC_BO_FROM_DEVICE
    );


    double squared_error_sum = 0.0;
    float max_absolute_error = 0.0f;

    for (int i = 0;
         i < output_size;
         i++)
    {
        const float difference =
            output_map[i]
            - output_ref[i];

        const float absolute_error =
            std::fabs(difference);

        squared_error_sum +=
            static_cast<double>(difference)
            * static_cast<double>(difference);

        if (absolute_error >
            max_absolute_error)
        {
            max_absolute_error =
                absolute_error;
        }
    }


    const double rmse =
        std::sqrt(
            squared_error_sum
            / static_cast<double>(
                output_size
            )
        );


    std::printf(
        "\n%s\n",
        test_name
    );

    std::printf(
        "Input size  = %d\n",
        input_size
    );

    std::printf(
        "Output size = %d\n",
        output_size
    );

    std::printf(
        "RMSE        = %.8e\n",
        rmse
    );

    std::printf(
        "Max error   = %.8e\n",
        max_absolute_error
    );


    const bool passed =
        (rmse <= EPSILON)
        &&
        (max_absolute_error <= EPSILON);


    std::printf(
        "Result      = %s\n",
        passed ? "PASS" : "FAIL"
    );


    return passed;
}


int main()
{
    std::cout
        << "Abriendo dispositivo..."
        << std::endl;

    xrt::device device(0);


    std::cout
        << "Cargando xclbin..."
        << std::endl;

    auto uuid =
        device.load_xclbin(
            "/lib/firmware/xilinx/avgpool2d/avgpool2d.xclbin"
        );


    std::cout
        << "Abriendo kernel avg_pool2d_forward..."
        << std::endl;

    xrt::kernel kernel(
        device,
        uuid,
        "avg_pool2d_forward"
    );


    bool all_passed = true;


    all_passed &=
        run_test(
            device,
            kernel,
            "Test 1 - LeNet AvgPool2D 6x28x28 -> 6x14x14",
            64,
            6,
            28,
            28,
            14,
            14,
            2,
            2,
            2,
            2,
            0,
            0,
            false,
            false,
            0
        );


    all_passed &=
        run_test(
            device,
            kernel,
            "Test 2 - LeNet AvgPool2D 16x10x10 -> 16x5x5",
            64,
            16,
            10,
            10,
            5,
            5,
            2,
            2,
            2,
            2,
            0,
            0,
            false,
            false,
            0
        );


    all_passed &=
        run_test(
            device,
            kernel,
            "Test 3 - Padding, count_include_pad=false",
            1,
            1,
            3,
            3,
            4,
            4,
            2,
            2,
            1,
            1,
            1,
            1,
            false,
            false,
            0
        );


    all_passed &=
        run_test(
            device,
            kernel,
            "Test 4 - Padding, count_include_pad=true",
            1,
            1,
            3,
            3,
            4,
            4,
            2,
            2,
            1,
            1,
            1,
            1,
            true,
            false,
            0
        );


    if (all_passed)
    {
        std::cout
            << "\nALL AVGPOOL2D TESTS PASSED"
            << std::endl;

        return 0;
    }


    std::cout
        << "\nAVGPOOL2D TEST FAILED"
        << std::endl;

    return 1;
}
