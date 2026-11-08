#include "fpga_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int main()
{
    try {
        auto& runtime = FpgaRuntime::instance();

        // ----------------------------------------------------
        // Dimensiones del grafo compartido por todos los batches
        // ----------------------------------------------------
        constexpr int batch_size = 16;
        constexpr std::size_t batch_count = 2;
        constexpr int input_channels = 1;
        constexpr int input_height = 8;
        constexpr int input_width = 8;
        constexpr int output_channels = 2;
        constexpr int kernel_size = 5;
        constexpr int padding = 2;
        constexpr int stride = 1;
        constexpr int output_height = 8;
        constexpr int output_width = 8;
        constexpr int pool_kernel = 2;
        constexpr int pool_stride = 2;
        constexpr int pool_height =
            (output_height - pool_kernel) / pool_stride + 1;
        constexpr int pool_width =
            (output_width - pool_kernel) / pool_stride + 1;
        constexpr int in_features =
            output_channels * pool_height * pool_width;
        constexpr int out_features = 64;
        constexpr int depthwise_kernel = 3;
        constexpr float bn_eps = 1e-5f;
        constexpr float residual_alpha = 1.0f;
        constexpr int input_count =
            batch_size * input_channels * input_height * input_width;
        constexpr int conv_count =
            batch_size * output_channels * output_height * output_width;
        constexpr int pool_count = batch_size * in_features;
        constexpr int output_count = batch_size * out_features;
        constexpr std::size_t input_bytes =
            static_cast<std::size_t>(input_count) * sizeof(float);
        constexpr std::size_t expected_nodes = 9;

        // ----------------------------------------------------
        // Parámetros deterministas, compartidos y persistentes
        // ----------------------------------------------------
        std::vector<float> conv_weights(
            output_channels * input_channels * kernel_size * kernel_size);
        std::vector<float> conv_bias(output_channels);
        std::vector<float> linear_weights(out_features * in_features);
        std::vector<float> linear_bias(out_features);
        std::vector<float> pointwise_weights(output_channels * output_channels);
        std::vector<float> pointwise_bias(output_channels);
        std::vector<float> depthwise_weights(
            output_channels * depthwise_kernel * depthwise_kernel);
        std::vector<float> depthwise_bias(output_channels);
        std::vector<float> bn_weight(output_channels);
        std::vector<float> bn_bias(output_channels);
        std::vector<float> bn_mean(output_channels);
        std::vector<float> bn_variance(output_channels);

        for (std::size_t i = 0; i < conv_weights.size(); ++i)
            conv_weights[i] = (static_cast<int>(i % 11) - 5) * 0.02f;
        for (std::size_t i = 0; i < linear_weights.size(); ++i)
            linear_weights[i] = (static_cast<int>(i % 19) - 9) * 0.01f;
        for (int i = 0; i < out_features; ++i)
            linear_bias[i] = (i % 9 - 4) * 0.05f;
        for (std::size_t i = 0; i < pointwise_weights.size(); ++i)
            pointwise_weights[i] = (static_cast<int>(i % 7) - 3) * 0.08f;
        for (std::size_t i = 0; i < depthwise_weights.size(); ++i)
            depthwise_weights[i] = (static_cast<int>(i % 9) - 4) * 0.03f;
        for (int c = 0; c < output_channels; ++c) {
            conv_bias[c] = (c + 1) * 0.03f;
            pointwise_bias[c] = (c + 1) * 0.02f;
            depthwise_bias[c] = (c - 1) * 0.015f;
            bn_weight[c] = 0.8f + c * 0.1f;
            bn_bias[c] = (c - 1) * 0.03f;
            bn_mean[c] = (c + 1) * 0.04f;
            bn_variance[c] = 0.7f + c * 0.2f;
        }

        struct Parameter {
            std::string id;
            std::vector<float>* values;
            int memory_group;
        };
        const int input_group =
            runtime.get_kernel("conv2d_forward").group_id(0);
        const std::vector<Parameter> parameters = {
            {"conv_weights", &conv_weights,
                runtime.get_kernel("conv2d_forward").group_id(1)},
            {"conv_bias", &conv_bias,
                runtime.get_kernel("conv2d_forward").group_id(2)},
            {"linear_weights", &linear_weights,
                runtime.get_kernel("linear_forward").group_id(1)},
            {"linear_bias", &linear_bias,
                runtime.get_kernel("linear_forward").group_id(2)},
            {"pointwise_weights", &pointwise_weights,
                runtime.get_kernel("pointwise_conv2d_forward").group_id(1)},
            {"pointwise_bias", &pointwise_bias,
                runtime.get_kernel("pointwise_conv2d_forward").group_id(2)},
            {"depthwise_weights", &depthwise_weights,
                runtime.get_kernel("depthwise_conv2d_forward").group_id(1)},
            {"depthwise_bias", &depthwise_bias,
                runtime.get_kernel("depthwise_conv2d_forward").group_id(2)},
            {"bn_weight", &bn_weight,
                runtime.get_kernel("batchnorm2d_forward").group_id(1)},
            {"bn_bias", &bn_bias,
                runtime.get_kernel("batchnorm2d_forward").group_id(2)},
            {"bn_mean", &bn_mean,
                runtime.get_kernel("batchnorm2d_forward").group_id(3)},
            {"bn_variance", &bn_variance,
                runtime.get_kernel("batchnorm2d_forward").group_id(4)}
        };
        for (const auto& parameter : parameters) {
            runtime.load_parameter(parameter.id, parameter.values->data(),
                parameter.values->size(), parameter.memory_group);
        }

        // ----------------------------------------------------
        // Construcción del grafo con una conexión residual
        // Pointwise tiene dos consumidores: Depthwise y LayerAdd.
        // ----------------------------------------------------
        runtime.clear_graph();
        runtime.add_graph_input("input");
        for (const auto& parameter : parameters)
            runtime.add_graph_parameter(parameter.id);

        OperationNode conv;
        conv.operation = "Conv2D";
        conv.inputs = {"input"};
        conv.parameters = {"conv_weights", "conv_bias"};
        conv.output = "conv_output";
        conv.int_args = {
            {"batch_size", batch_size}, {"input_channels", input_channels},
            {"input_height", input_height}, {"input_width", input_width},
            {"output_channels", output_channels},
            {"output_height", output_height}, {"output_width", output_width},
            {"kernel_height", kernel_size}, {"kernel_width", kernel_size},
            {"stride_height", stride}, {"stride_width", stride},
            {"padding_height", padding}, {"padding_width", padding},
            {"use_bias", 1}
        };
        runtime.add_operation(conv);

        OperationNode pointwise;
        pointwise.operation = "Pointwise";
        pointwise.inputs = {"conv_output"};
        pointwise.parameters = {"pointwise_weights", "pointwise_bias"};
        pointwise.output = "pointwise_output";
        pointwise.int_args = {
            {"batch_size", batch_size}, {"input_channels", output_channels},
            {"input_height", output_height}, {"input_width", output_width},
            {"output_channels", output_channels}, {"use_bias", 1}
        };
        runtime.add_operation(pointwise);

        OperationNode depthwise;
        depthwise.operation = "Depthwise";
        depthwise.inputs = {"pointwise_output"};
        depthwise.parameters = {"depthwise_weights", "depthwise_bias"};
        depthwise.output = "depthwise_output";
        depthwise.int_args = {
            {"batch_size", batch_size}, {"channels", output_channels},
            {"input_height", output_height}, {"input_width", output_width},
            {"output_height", output_height}, {"output_width", output_width},
            {"kernel_height", depthwise_kernel},
            {"kernel_width", depthwise_kernel},
            {"stride_height", 1}, {"stride_width", 1},
            {"padding_height", 1}, {"padding_width", 1}, {"use_bias", 1}
        };
        runtime.add_operation(depthwise);

        OperationNode bn;
        bn.operation = "BatchNorm2D";
        bn.inputs = {"depthwise_output"};
        bn.parameters = {"bn_weight", "bn_bias", "bn_mean", "bn_variance"};
        bn.output = "bn_output";
        bn.int_args = {
            {"batch_size", batch_size}, {"channels", output_channels},
            {"input_height", output_height}, {"input_width", output_width},
            {"affine", 1}, {"use_bias", 1}
        };
        bn.float_args["eps"] = bn_eps;
        runtime.add_operation(bn);

        OperationNode add;
        add.operation = "LayerAdd";
        add.inputs = {"bn_output", "pointwise_output"};
        add.output = "residual_output";
        add.int_args["size"] = conv_count;
        add.float_args["alpha"] = residual_alpha;
        runtime.add_operation(add);

        OperationNode pool;
        pool.operation = "AvgPool2D";
        pool.inputs = {"residual_output"};
        pool.output = "pool_output";
        pool.int_args = {
            {"batch_size", batch_size}, {"input_channels", output_channels},
            {"input_height", output_height}, {"input_width", output_width},
            {"output_height", pool_height}, {"output_width", pool_width},
            {"kernel_height", pool_kernel}, {"kernel_width", pool_kernel},
            {"stride_height", pool_stride}, {"stride_width", pool_stride},
            {"padding_height", 0}, {"padding_width", 0},
            {"count_include_pad", 0}, {"use_divisor_override", 0},
            {"divisor_override", 0}
        };
        runtime.add_operation(pool);

        OperationNode linear;
        linear.operation = "Linear";
        linear.inputs = {"pool_output"};
        linear.parameters = {"linear_weights", "linear_bias"};
        linear.output = "linear_output";
        linear.int_args = {
            {"batch_size", batch_size}, {"in_features", in_features},
            {"out_features", out_features}, {"use_bias", 1}
        };
        runtime.add_operation(linear);

        OperationNode relu;
        relu.operation = "ReLU6";
        relu.inputs = {"linear_output"};
        relu.output = "relu_output";
        relu.int_args["size"] = output_count;
        runtime.add_operation(relu);

        OperationNode tanh;
        tanh.operation = "Tanh";
        tanh.inputs = {"relu_output"};
        tanh.output = "output";
        tanh.int_args["size"] = output_count;
        runtime.add_operation(tanh);
        runtime.set_graph_outputs({"output"});
        runtime.finalize_graph();

        if (runtime.graph_nodes().size() != expected_nodes)
            throw std::runtime_error("El test debe registrar nueve nodos");
        if (runtime.graph_consumers("pointwise_output") != 2)
            throw std::runtime_error("La conexion residual requiere dos consumidores");

        std::cout << "Nodos registrados: " << runtime.graph_nodes().size() << '\n';
        for (const auto& node : runtime.graph_nodes())
            std::cout << "  " << node.operation << '\n';

        // Obtener los kernels restantes antes de medir.
        runtime.get_kernel("avg_pool2d_forward");
        runtime.get_kernel("layer_add_forward");
        runtime.get_kernel("relu6_forward");
        runtime.get_kernel("tanh_forward");

        // ----------------------------------------------------
        // Entradas distintas para cada batch
        // Con dos batches se conservan las entradas originales.
        // ----------------------------------------------------
        std::vector<std::vector<float>> data(
            batch_count, std::vector<float>(input_count));
        std::vector<std::vector<double>> expected(
            batch_count, std::vector<double>(output_count));
        for (std::size_t b = 0; b < batch_count; ++b) {
            for (int i = 0; i < input_count; ++i) {
                if (b == 0)
                    data[b][i] = (i % 201 - 100) / 10.0f;
                else
                    data[b][i] = ((i * 7 + static_cast<int>(b - 1) * 13)
                        % 251 - 125) / 12.0f;
            }
        }

        // ----------------------------------------------------
        // Referencia CPU fuera del cronómetro
        // Cada interfaz entre operaciones se redondea a float32.
        // ----------------------------------------------------
        const auto index_nchw = [](int n, int c, int h, int w) {
            return ((n * output_channels + c) * output_height + h)
                * output_width + w;
        };
        for (std::size_t b = 0; b < batch_count; ++b) {
            std::vector<float> conv_reference(conv_count);
            for (int n = 0; n < batch_size; ++n)
                for (int oc = 0; oc < output_channels; ++oc)
                    for (int oh = 0; oh < output_height; ++oh)
                        for (int ow = 0; ow < output_width; ++ow) {
                            double sum = conv_bias[oc];
                            for (int ic = 0; ic < input_channels; ++ic)
                                for (int kh = 0; kh < kernel_size; ++kh)
                                    for (int kw = 0; kw < kernel_size; ++kw) {
                                        const int ih = oh * stride - padding + kh;
                                        const int iw = ow * stride - padding + kw;
                                        if (ih < 0 || ih >= input_height ||
                                            iw < 0 || iw >= input_width)
                                            continue;
                                        const int input_index =
                                            ((n * input_channels + ic) * input_height
                                                + ih) * input_width + iw;
                                        const int weight_index =
                                            ((oc * input_channels + ic) * kernel_size
                                                + kh) * kernel_size + kw;
                                        sum += static_cast<double>(data[b][input_index])
                                            * conv_weights[weight_index];
                                    }
                            conv_reference[index_nchw(n, oc, oh, ow)] =
                                static_cast<float>(sum);
                        }

            // Convolución 1x1: pesos con orden salida, entrada.
            std::vector<float> pointwise_reference(conv_count);
            for (int n = 0; n < batch_size; ++n)
                for (int oc = 0; oc < output_channels; ++oc)
                    for (int h = 0; h < output_height; ++h)
                        for (int w = 0; w < output_width; ++w) {
                            double sum = pointwise_bias[oc];
                            for (int ic = 0; ic < output_channels; ++ic)
                                sum += static_cast<double>(
                                    conv_reference[index_nchw(n, ic, h, w)])
                                    * pointwise_weights[oc * output_channels + ic];
                            pointwise_reference[index_nchw(n, oc, h, w)] =
                                static_cast<float>(sum);
                        }

            // Convolución espacial independiente por canal.
            std::vector<float> depthwise_reference(conv_count);
            for (int n = 0; n < batch_size; ++n)
                for (int c = 0; c < output_channels; ++c)
                    for (int h = 0; h < output_height; ++h)
                        for (int w = 0; w < output_width; ++w) {
                            double sum = depthwise_bias[c];
                            for (int kh = 0; kh < depthwise_kernel; ++kh)
                                for (int kw = 0; kw < depthwise_kernel; ++kw) {
                                    const int ih = h - 1 + kh;
                                    const int iw = w - 1 + kw;
                                    if (ih < 0 || ih >= output_height ||
                                        iw < 0 || iw >= output_width)
                                        continue;
                                    const int weight_index =
                                        (c * depthwise_kernel + kh)
                                            * depthwise_kernel + kw;
                                    sum += static_cast<double>(pointwise_reference[
                                        index_nchw(n, c, ih, iw)])
                                        * depthwise_weights[weight_index];
                                }
                            depthwise_reference[index_nchw(n, c, h, w)] =
                                static_cast<float>(sum);
                        }

            // Normalización con estadísticas de inferencia y suma residual.
            std::vector<float> residual_reference(conv_count);
            for (int n = 0; n < batch_size; ++n)
                for (int c = 0; c < output_channels; ++c) {
                    const double denominator = std::sqrt(
                        static_cast<double>(bn_variance[c]) + bn_eps);
                    for (int h = 0; h < output_height; ++h)
                        for (int w = 0; w < output_width; ++w) {
                            const int index = index_nchw(n, c, h, w);
                            const float normalized = static_cast<float>(
                                (static_cast<double>(depthwise_reference[index])
                                    - bn_mean[c]) / denominator
                                    * bn_weight[c] + bn_bias[c]);
                            residual_reference[index] = normalized
                                + residual_alpha * pointwise_reference[index];
                        }
                }

            // Promedio de ventanas 2x2, sin padding.
            std::vector<float> pool_reference(pool_count);
            for (int n = 0; n < batch_size; ++n)
                for (int c = 0; c < output_channels; ++c)
                    for (int ph = 0; ph < pool_height; ++ph)
                        for (int pw = 0; pw < pool_width; ++pw) {
                            double sum = 0.0;
                            for (int kh = 0; kh < pool_kernel; ++kh)
                                for (int kw = 0; kw < pool_kernel; ++kw)
                                    sum += residual_reference[index_nchw(n, c,
                                        ph * pool_stride + kh, pw * pool_stride + kw)];
                            const int index = ((n * output_channels + c)
                                * pool_height + ph) * pool_width + pw;
                            pool_reference[index] = static_cast<float>(
                                sum / (pool_kernel * pool_kernel));
                        }

            // Aplanado NCHW por muestra, Linear, ReLU6 y Tanh.
            for (int n = 0; n < batch_size; ++n)
                for (int out = 0; out < out_features; ++out) {
                    double sum = linear_bias[out];
                    for (int in = 0; in < in_features; ++in)
                        sum += static_cast<double>(pool_reference[n * in_features + in])
                            * linear_weights[out * in_features + in];
                    const float linear_value = static_cast<float>(sum);
                    const float clipped = std::min(6.0f, std::max(0.0f, linear_value));
                    expected[b][n * out_features + out] =
                        static_cast<float>(std::tanh(clipped));
                }
        }

        std::vector<double> sequential_ms;
        std::vector<double> pipeline_ms;
        sequential_ms.reserve(5);
        pipeline_ms.reserve(5);
        std::cout
            << "Grafo: Conv2D -> Pointwise -> Depthwise -> BatchNorm2D"
            << " -> LayerAdd -> AvgPool2D -> Linear -> ReLU6 -> Tanh\n"
            << "Residual: Pointwise -> segunda entrada de LayerAdd\n"
            << "Batches independientes: " << batch_count << '\n'
            << "Muestras por batch: " << batch_size << '\n';

        // ----------------------------------------------------
        // Un calentamiento y cinco mediciones por modo
        // Secuencial usa el mismo planificador con un batch a la vez.
        // No se compara aquí contra execute_graph().
        // ----------------------------------------------------
        for (int repetition = 0; repetition < 12; ++repetition) {
            const bool warmup = repetition < 2;
            const bool pipeline = (repetition / 2) % 2 == 0
                ? repetition % 2 == 1 : repetition % 2 == 0;
            std::vector<PipelineBatchState> batches(batch_count);

            // Transferir entradas antes de iniciar el cronómetro.
            for (std::size_t b = 0; b < batch_count; ++b) {
                const std::string id = "test_input_"
                    + std::to_string(repetition) + "_" + std::to_string(b);
                auto& input = runtime.create_buffer(id, input_bytes, input_group);
                std::copy(data[b].begin(), data[b].end(), input.map<float*>());
                input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                batches[b].tensor_buffers.emplace("input", id);
            }
            std::vector<std::vector<PipelineBatchState>> sequential_batches;
            if (!pipeline) {
                sequential_batches.resize(batch_count);
                for (std::size_t b = 0; b < batch_count; ++b)
                    sequential_batches[b].push_back(batches[b]);
            }

            // Incluir planificación, memoria interna y ejecución.
            const auto start = std::chrono::steady_clock::now();
            if (pipeline)
                runtime.execute_pipeline(batches);
            else
                for (auto& single_batch : sequential_batches)
                    runtime.execute_pipeline(single_batch);
            const auto finish = std::chrono::steady_clock::now();
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(finish - start).count();
            if (!pipeline)
                for (std::size_t b = 0; b < batch_count; ++b)
                    batches[b] = std::move(sequential_batches[b][0]);

            std::cout << (warmup ? "Calentamiento" : "Medicion")
                << " | " << (pipeline ? "Pipeline" : "Secuencial")
                << " | " << batch_count << " batches: " << elapsed_ms << " ms\n";

            // ------------------------------------------------
            // Descargar y validar fuera del tiempo medido
            // ------------------------------------------------
            for (std::size_t b = 0; b < batch_count; ++b) {
                if (batches[b].completed_nodes != expected_nodes ||
                    batches[b].node_states.size() != expected_nodes)
                    throw std::runtime_error("El batch no completo todos los nodos");
                for (const auto state : batches[b].node_states)
                    if (state != PipelineNodeState::Completed)
                        throw std::runtime_error("Un nodo no termino correctamente");
                const std::string output_id = batches[b].tensor_buffers.at("output");
                auto& output = runtime.buffer(output_id);
                output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                const auto* result = output.map<float*>();
                double max_error = 0.0;
                for (int i = 0; i < output_count; ++i) {
                    if (!std::isfinite(result[i]))
                        throw std::runtime_error("La salida contiene un valor no finito");
                    max_error = std::max(max_error,
                        std::abs(static_cast<double>(result[i]) - expected[b][i]));
                }
                runtime.consume_buffer(output_id);
                std::cout << "  Batch " << b << " | Error maximo: " << max_error << '\n';
                if (max_error > 1e-5)
                    throw std::runtime_error("La salida no coincide con la referencia");
            }

            // Solo deben permanecer los parámetros persistentes.
            if (runtime.buffer_count() != parameters.size())
                throw std::runtime_error("Cantidad incorrecta de buffers activos");
            for (const auto& parameter : parameters) {
                auto& bo = runtime.buffer(parameter.id);
                bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                if (!std::equal(parameter.values->begin(), parameter.values->end(),
                        bo.map<float*>()))
                    throw std::runtime_error("Parametro modificado: " + parameter.id);
            }
            if (!warmup) {
                if (pipeline) pipeline_ms.push_back(elapsed_ms);
                else sequential_ms.push_back(elapsed_ms);
            }
            std::cout << "  Reutilizaciones acumuladas: "
                << runtime.reuse_count() << '\n';
        }

        // ----------------------------------------------------
        // Liberación y resumen de mediciones válidas
        // ----------------------------------------------------
        for (const auto& parameter : parameters)
            runtime.release_buffer(parameter.id);
        if (runtime.buffer_count() != 0)
            throw std::runtime_error("Quedaron buffers activos al finalizar");
        if (sequential_ms.size() != 5 || pipeline_ms.size() != 5)
            throw std::runtime_error("Se requieren cinco mediciones por modo");
        const auto mean = [](const std::vector<double>& values) {
            double total = 0.0;
            for (const double value : values) total += value;
            return total / values.size();
        };
        const double seq_mean = mean(sequential_ms);
        const double pipe_mean = mean(pipeline_ms);
        std::cout << "Iniciando limpieza XRT..." << std::endl;
        runtime.shutdown();
        std::cout << "Limpieza XRT completada." << std::endl;
        std::cout << "\nPromedio secuencial: " << seq_mean << " ms"
            << "\nPromedio pipeline:   " << pipe_mean << " ms"
            << "\nAceleracion: " << seq_mean / pipe_mean << "x\n"
            << "\nPASS: resultados, parametros y buffers correctos\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}