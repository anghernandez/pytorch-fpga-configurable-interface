"""Registro diferido de operaciones FPGA para inferencia (sin autograd).

Una instancia de FpgaGraph representa un solo DAG activo en FpgaRuntime.
Los pesos se transfieren al construir el grafo y se mantienen persistentes.
"""
import numpy as np
import torch
import fpga_dag


KERNELS = {
    "ReLU6": "relu6_forward", "Tanh": "tanh_forward",
    "Linear": "linear_forward", "Conv2D": "conv2d_forward",
    "AvgPool2D": "avg_pool2d_forward", "Pointwise": "pointwise_conv2d_forward",
    "Depthwise": "depthwise_conv2d_forward", "BatchNorm2D": "batchnorm2d_forward",
    "LayerAdd": "layer_add_forward",
}


class FpgaTensor:
    """Referencia simbólica (identificador y forma, no datos en CPU)."""
    def __init__(self, graph, tensor_id, shape):
        self.graph = graph
        self.tensor_id = str(tensor_id)
        self.shape = tuple(int(dim) for dim in shape)

    def reshape(self, *shape):
        """Cambia únicamente la forma simbólica; conserva el buffer XRT."""
        self.graph._check_tensor(self)

        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = tuple(shape[0])

        if not shape:
            raise ValueError("reshape necesita una forma de salida")
        if any(not isinstance(dim, int) or isinstance(dim, bool) for dim in shape):
            raise TypeError("Las dimensiones de reshape deben ser enteros")

        source_elements = self.graph.element_count(self.shape)
        inferred = [i for i, dim in enumerate(shape) if dim == -1]
        if len(inferred) > 1:
            raise ValueError("reshape solo admite una dimensión -1")
        if any(dim == 0 or dim < -1 for dim in shape):
            raise ValueError("reshape requiere dimensiones positivas o una -1")

        target = list(shape)
        if inferred:
            known = self.graph.element_count(dim for dim in target if dim != -1)
            if source_elements % known != 0:
                raise ValueError("reshape incompatible con el número de elementos")
            target[inferred[0]] = source_elements // known

        if self.graph.element_count(target) != source_elements:
            raise ValueError("reshape no puede cambiar el número de elementos")

        return FpgaTensor(self.graph, self.tensor_id, target)


class FpgaGraph:
    def __init__(self):
        self._input_id = "graph_input_0"
        self._output = None
        self._shape = None
        self._counter = 0
        self._compiled = False
        self._first_kernel = None
        self._first_argument = 0
        self._registered = set()
        self._pending_parameters = []
        self._parameter_counter = 0

    @staticmethod
    def element_count(shape):
        result = 1
        for dimension in shape:
            if int(dimension) <= 0:
                raise ValueError("Dimensión no positiva")
            result *= int(dimension)
        return result

    def _check_tensor(self, tensor):
        if not isinstance(tensor, FpgaTensor) or tensor.graph is not self:
            raise TypeError("Se requiere FpgaTensor del grafo activo")
        if self._compiled:
            raise RuntimeError("El DAG ya se finalizó")

    def add_node(self, operation, inputs, parameters, output_shape,
                 int_args=None, float_args=None):
        for tensor in inputs:
            self._check_tensor(tensor)
        self.element_count(output_shape)
        if operation not in KERNELS:
            raise ValueError(f"Operación FPGA desconocida: {operation}")
        if self._first_kernel is None:
            self._first_kernel = KERNELS[operation]
            self._first_argument = 0
        self._counter += 1
        output_id = f"graph_tensor_{self._counter}"
        fpga_dag.add_operation(operation, [t.tensor_id for t in inputs],
                               list(parameters), output_id,
                               dict(int_args or {}), dict(float_args or {}))
        return FpgaTensor(self, output_id, output_shape)

    def add_unary(self, operation, tensor):
        return self.add_node(operation, [tensor], [], tensor.shape,
                             {"size": self.element_count(tensor.shape)})

    def register_parameter(self, module, role, data, kernel_name, argument_index):
        if self._compiled:
            raise RuntimeError("No se pueden registrar parámetros tras build()")
        # Mismo objeto+cargo dentro de un DAG: reutilizar referencia.
        key = (id(module), role)
        if key in self._registered:
            return next(item[0] for item in self._pending_parameters if item[4] == key)
        self._parameter_counter += 1
        name = f"fpga_param_{self._parameter_counter}"
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        array = np.ascontiguousarray(data, dtype=np.float32)
        if array.size == 0:
            raise ValueError("Parámetro vacío")
        fpga_dag.add_parameter(name)
        self._registered.add(key)
        self._pending_parameters.append((name, array, kernel_name, argument_index, key))
        return name

    def register_weight_bias(self, module, kernel_name, weight_index, bias_index, bias_count):
        weights = self.register_parameter(module, "weight", module.weight,
                                          kernel_name, weight_index)
        bias = module.bias
        if bias is None:
            bias = np.zeros(bias_count, dtype=np.float32)
        biases = self.register_parameter(module, "bias", bias,
                                         kernel_name, bias_index)
        return [weights, biases]

    def build(self, model, example_input):
        if not isinstance(example_input, torch.Tensor):
            raise TypeError("example_input debe ser torch.Tensor")
        if example_input.dtype != torch.float32:
            raise TypeError("Solo float32")
        self._shape = tuple(example_input.shape)
        self.element_count(self._shape)
        self._compiled = False
        self._counter = self._parameter_counter = 0
        self._first_kernel = None
        self._registered.clear()
        self._pending_parameters.clear()
        fpga_dag.clear_graph()
        fpga_dag.add_input(self._input_id)
        output = model(FpgaTensor(self, self._input_id, self._shape))
        if not isinstance(output, FpgaTensor) or output.graph is not self:
            raise TypeError("El modelo debe retornar un único FpgaTensor")
        if self._counter == 0:
            raise ValueError("El grafo está vacío")
        fpga_dag.set_outputs([output.tensor_id])
        fpga_dag.finalize_graph()
        # En carga inicial, pesos y estadísticas pasan CPU->XRT una sola vez.
        for name, array, kernel, index, _ in self._pending_parameters:
            fpga_dag.load_parameter(name, array, kernel, index)
        self._output = output
        self._compiled = True
        return self

    def run(self, input_tensor):
        if not self._compiled:
            raise RuntimeError("Primero llame build()")
        if not isinstance(input_tensor, torch.Tensor) or tuple(input_tensor.shape) != self._shape:
            raise ValueError("La entrada no coincide con la forma compilada")
        if input_tensor.dtype != torch.float32 or input_tensor.requires_grad:
            raise ValueError("Se requiere float32 sin gradientes")
        data = np.ascontiguousarray(input_tensor.detach().cpu().numpy())
        fpga_dag.upload_input(self._input_id, data, self._first_kernel, self._first_argument)
        fpga_dag.execute_graph()
        result = fpga_dag.download_output(self._output.tensor_id, list(self._output.shape))
        return torch.from_numpy(result).to(input_tensor.device)