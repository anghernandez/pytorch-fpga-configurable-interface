import numpy as np
import torch
import fpga_dag

import time

# Relaciona las operaciones del grafo con sus kernels XRT
KERNELS = {
    "ReLU6": "relu6_forward", "Tanh": "tanh_forward",
    "Linear": "linear_forward", "Conv2D": "conv2d_forward",
    "AvgPool2D": "avg_pool2d_forward", "Pointwise": "pointwise_conv2d_forward",
    "Depthwise": "depthwise_conv2d_forward", "BatchNorm2D": "batchnorm2d_forward",
    "LayerAdd": "layer_add_forward",
}

# ------------------------------------------------------------
# Representa un tensor simbólico mediante su ID y forma.
# No almacena directamente los datos de las activaciones.
# ------------------------------------------------------------
class FpgaTensor:

    # Asocia el tensor con su grafo, identificador y dimensiones.
    def __init__(self, graph, tensor_id, shape):
        self.graph = graph
        self.tensor_id = str(tensor_id)
        self.shape = tuple(int(dim) for dim in shape)

    # Modifica la forma simbólica sin crear ni copiar buffers XRT
    def reshape(self, *shape):
        self.graph._check_tensor(self)

        # Calcula la dimensión -1 a partir del número de elementos.
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

        # Conserva el ID original y devuelve una nueva referencia simbólica
        return FpgaTensor(self.graph, self.tensor_id, target)


# ------------------------------------------------------------
# Construye y ejecuta el DAG mediante el módulo fpga_dag
# Administra las referencias simbólicas y los parámetros
# ------------------------------------------------------------
class FpgaGraph:

    # Inicializa el estado necesario para construir el grafo
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
        # Generar identificadores únicos para las entradas del pipeline.
        self._pipeline_counter = 0

    # Calcula el número de elementos a partir de las dimensiones
    @staticmethod
    def element_count(shape):
        result = 1
        for dimension in shape:
            if int(dimension) <= 0:
                raise ValueError("Dimensión no positiva")
            result *= int(dimension)
        return result

    # Verifica que el tensor pertenezca al grafo activo
    def _check_tensor(self, tensor):
        if not isinstance(tensor, FpgaTensor) or tensor.graph is not self:
            raise TypeError("Se requiere FpgaTensor del grafo activo")
        if self._compiled:
            raise RuntimeError("El DAG ya se finalizó")

    # ------------------------------------------------------------
    # Registra una operación en el DAG y devuelve su tensor simbólico
    # No ejecuta el kernel.
    # ------------------------------------------------------------
    def add_node(self, operation, inputs, parameters, output_shape,
                 int_args=None, float_args=None):
        for tensor in inputs:
            self._check_tensor(tensor)
        self.element_count(output_shape)
        if operation not in KERNELS:
            raise ValueError(f"Operación FPGA desconocida: {operation}")

        # Identifica el kernel que recibirá la entrada inicial del grafo
        if self._first_kernel is None:
            self._first_kernel = KERNELS[operation]
            self._first_argument = 0
        self._counter += 1
        output_id = f"graph_tensor_{self._counter}"

        # Envía la descripción del nodo al runtime mediante pybind11
        fpga_dag.add_operation(operation, [t.tensor_id for t in inputs],
                               list(parameters), output_id,
                               dict(int_args or {}), dict(float_args or {}))
        return FpgaTensor(self, output_id, output_shape)

    # Registra una operación unaria conservando la forma del tensor
    def add_unary(self, operation, tensor):
        return self.add_node(operation, [tensor], [], tensor.shape,
                             {"size": self.element_count(tensor.shape)})

    # ------------------------------------------------------------
    # Registra un parámetro y prepara sus datos para cargarlos en XRT.
    # Evita registrar dos veces el mismo parámetro del módulo.
    # ------------------------------------------------------------
    def register_parameter(self, module, role, data, kernel_name, argument_index):
        if self._compiled:
            raise RuntimeError("No se pueden registrar parámetros tras build()")
        
        # Identifica el parámetro mediante su módulo y función
        key = (id(module), role)
        if key in self._registered:
            return next(item[0] for item in self._pending_parameters if item[4] == key)
        self._parameter_counter += 1
        name = f"fpga_param_{self._parameter_counter}"
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        # Convierte el parámetro a un arreglo NumPy float32 contiguo
        array = np.ascontiguousarray(data, dtype=np.float32)
        if array.size == 0:
            raise ValueError("Parámetro vacío")
        fpga_dag.add_parameter(name)
        self._registered.add(key)

        # Conserva los datos para cargarlos durante build()
        self._pending_parameters.append((name, array, kernel_name, argument_index, key))
        return name

    # Registra pesos y bias; genera un bias de ceros si no existe
    def register_weight_bias(self, module, kernel_name, weight_index, bias_index, bias_count):
        weights = self.register_parameter(module, "weight", module.weight,
                                          kernel_name, weight_index)
        bias = module.bias
        if bias is None:
            bias = np.zeros(bias_count, dtype=np.float32)
        biases = self.register_parameter(module, "bias", bias,
                                         kernel_name, bias_index)
        return [weights, biases]

    # ------------------------------------------------------------
    # Construye el DAG a partir del modelo, valida sus dependencias
    # y carga los parámetros persistentes en la FPGA
    # ------------------------------------------------------------
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

        # Reinicia el DAG y registra su entrada simbólica
        fpga_dag.clear_graph()
        fpga_dag.add_input(self._input_id)

        # Recorre el modelo con tensores simbólicos para registrar sus operaciones
        output = model(FpgaTensor(self, self._input_id, self._shape))
        if not isinstance(output, FpgaTensor) or output.graph is not self:
            raise TypeError("El modelo debe retornar un único FpgaTensor")
        if self._counter == 0:
            raise ValueError("El grafo está vacío")
        fpga_dag.set_outputs([output.tensor_id])

        # Valida las dependencias y establece el orden topológico
        fpga_dag.finalize_graph()

        # Transfiere los parámetros registrados a buffers persistentes
        for name, array, kernel, index, _ in self._pending_parameters:
            fpga_dag.load_parameter(name, array, kernel, index)
        self._output = output
        self._compiled = True
        return self

    # ------------------------------------------------------------
    # Ejecuta una inferencia completa mediante el DAG secuencial
    # Transfiere la entrada, ejecuta los kernels y recupera la salida
    # ------------------------------------------------------------
    def run(self, input_tensor):
        if not self._compiled:
            raise RuntimeError("Primero llame build()")

        if (
            not isinstance(input_tensor, torch.Tensor)
            or tuple(input_tensor.shape) != self._shape
        ):
            raise ValueError(
                "La entrada no coincide con la forma compilada"
            )

        if (
            input_tensor.dtype != torch.float32
            or input_tensor.requires_grad
        ):
            raise ValueError("Se requiere float32 sin gradientes")

        # Prepara la entrada PyTorch como un arreglo NumPy contiguo
        data = np.ascontiguousarray(
            input_tensor.detach().cpu().numpy()
        )

        # Transfiere los datos de entrada al buffer XRT.
        # 1. Carga la entrada de cada batch en la FPGA. (recorrido de ejecución)
        fpga_dag.upload_input(
            self._input_id,
            data,
            self._first_kernel,
            self._first_argument
        )

        # Solicita al runtime la ejecución secuencial del DAG.
        fpga_dag.execute_graph()

        # Recupera la salida de la FPGA como arreglo NumPy.
        result = fpga_dag.download_output(
            self._output.tensor_id,
            list(self._output.shape)
        )

        # Convierte el resultado a un tensor PyTorch
        return torch.from_numpy(result).to(input_tensor.device)


    def run_batches(self, input_batches, mode="pipeline"):
  
        # ----------------------------------------------------
        # Validar configuración y entradas antes de cargar datos
        # ----------------------------------------------------
        if not self._compiled:
            raise RuntimeError("Primero llame build()")

        if mode not in ("sequential", "pipeline"):
            raise ValueError(
                "mode debe ser 'sequential' o 'pipeline'"
            )

        # Evitar interpretar un tensor como una colección de batches.
        if isinstance(input_batches, torch.Tensor):
            raise TypeError(
                "Se requiere una colección de batches, no un único tensor"
            )

        batches = list(input_batches)

        if not batches:
            raise ValueError("Se requiere al menos un batch")

        for index, tensor in enumerate(batches):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"El batch {index} debe ser torch.Tensor"
                )

            if tuple(tensor.shape) != self._shape:
                raise ValueError(
                    f"Batch {index}: forma {tuple(tensor.shape)}, "
                    f"pero el grafo espera {self._shape}"
                )

            if tensor.dtype != torch.float32 or tensor.requires_grad:
                raise ValueError(
                    f"Batch {index}: se requiere float32 sin gradientes"
                )

        # Ejecuta cada batch de manera independiente mediante run()
        if mode == "sequential":
            return [self.run(tensor) for tensor in batches]

        # Genera identificadores únicos para las entradas del pipeline
        self._pipeline_counter += 1
        prefix = f"pipeline_input_{self._pipeline_counter}"

        input_ids = []
        batch_inputs = []

        try:
            
            # Prepara y transfiere las entradas independientes de cada batch
            for index, tensor in enumerate(batches):
                buffer_id = f"{prefix}_{index}"

                data = np.ascontiguousarray(
                    tensor.detach().cpu().numpy(),
                    dtype=np.float32,
                )

                # Evitar sobrescribir un buffer que siga activo
                if fpga_dag.has_buffer(buffer_id):
                    raise RuntimeError(
                        f"Identificador de entrada ocupado: {buffer_id}"
                    )

                # Registrar antes de cargar para poder limpiar
                # un BO que se haya creado durante una carga fallida.
                input_ids.append(buffer_id)

                # 1. Carga la entrada de cada batch en la FPGA (recorrido de ejecución pipeline)
                fpga_dag.upload_input(
                    buffer_id,
                    data,
                    self._first_kernel,
                    self._first_argument,
                )

                # Asocia la entrada simbólica del DAG con el buffer de cada batch
                # 2. Asocia la entrada del grafo con el buffer de este batch (recorrido ejecución pipeline)
                batch_inputs.append({
                    self._input_id: buffer_id,
                })

        except Exception:
            # Aún no se enviaron kernels al planificador.
            # Estas entradas no tienen consumidores configurados.
            for buffer_id in input_ids:
                if fpga_dag.has_buffer(buffer_id):
                    fpga_dag.release_buffer(buffer_id)
            raise

        # ----------------------------------------------------
        # Ejecutar el mismo grafo con todos los batches
        # ----------------------------------------------------
        # El runtime administra las activaciones temporales
        # y conserva los pesos y bias compartidos.
        output_maps = fpga_dag.execute_pipeline(batch_inputs) # 3. Solicita al runtime ejecutar los batches en pipeline (recorrido de ejecución pipeline).

        # ----------------------------------------------------
        # Descargar resultados conservando el orden de entrada
        # ----------------------------------------------------

        # Prepara la recuperación de resultados en el orden original
        results = []

        try:
            for tensor, outputs in zip(batches, output_maps):
                output_id = outputs[self._output.tensor_id]

                # 33. Descarga la salida del batch y libera su buffer al finalizar (finaliza recorrido ejecución Pipeline)
                array = fpga_dag.download_output(
                    output_id,
                    list(self._output.shape),
                )

                results.append(
                    torch.from_numpy(array).to(tensor.device)
                )

        finally:
            # Libera las salidas que permanezcan activas,
            # incluso si ocurre un error durante la descarga.
            for outputs in output_maps:
                for output_id in outputs.values():
                    if fpga_dag.has_buffer(output_id):
                        # Descuenta el consumo final y devuelve el BO
                        # al pool si no quedan consumidores.
                        fpga_dag.release_output(output_id)

        # Devuelve los resultados de los batches como tensores PyTorch.
        return results        


        