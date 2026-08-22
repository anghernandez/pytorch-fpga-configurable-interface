import math
import random


#Implementación de ManualLinear

class ManualLinear:
    """
    Capa lineal manual

    Calcula:

        y = xW^T + b
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True
    ):
        if in_features <= 0:
            raise ValueError(
                "in_features debe ser mayor que 0"
            )

        if out_features <= 0:
            raise ValueError(
                "out_features debe ser mayor que 0"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.reset_parameters()

    def reset_parameters(self):
        """
        Inicializa los pesos y el bias dentro del intervalo:

        [-1/sqrt(in_features), 1/sqrt(in_features)]
        """

        bound = 1.0 / math.sqrt(self.in_features)

        self.weight = []

        for output_index in range(self.out_features):
            neuron_weights = []

            for input_index in range(self.in_features):
                value = random.uniform(-bound, bound)
                neuron_weights.append(value)

            self.weight.append(neuron_weights)

        if self.use_bias:
            self.bias = []

            for output_index in range(self.out_features):
                value = random.uniform(-bound, bound)
                self.bias.append(value)
        else:
            self.bias = None

    def forward(self, x):
        """
        Ejecuta el forward de la capa.

        Entrada:
            x con forma:

            (batch_size, in_features)

        Salida:
            output con forma:

            (batch_size, out_features)
        """

        self._validate_input(x)

        output = []

        for sample_index in range(len(x)):
            sample = x[sample_index]
            sample_output = []

            for output_index in range(
                self.out_features
            ):
                accumulated_value = 0.0

                for input_index in range(
                    self.in_features
                ):
                    input_value = sample[
                        input_index
                    ]

                    weight_value = self.weight[
                        output_index
                    ][input_index]

                    accumulated_value += (
                        input_value * weight_value
                    )

                if self.bias is not None:
                    accumulated_value += self.bias[
                        output_index
                    ]

                sample_output.append(
                    accumulated_value
                )

            output.append(sample_output)

        return output

    def load_parameters(
        self,
        weight,
        bias=None
    ):
        """
        Carga pesos y bias externos.

        Esto permite copiar parámetros desde
        una capa nn.Linear de PyTorch.
        """

        self._validate_weight(weight)
        self._validate_bias(bias)

        self.weight = []

        for row in weight:
            self.weight.append(row.copy())

        if bias is not None:
            self.bias = bias.copy()
        else:
            self.bias = None

    def _validate_input(self, x):
        if not isinstance(x, list):
            raise TypeError(
                "La entrada x debe ser una lista"
            )

        if len(x) == 0:
            raise ValueError(
                "La entrada no puede estar vacía"
            )

        for sample_index in range(len(x)):
            sample = x[sample_index]

            if not isinstance(sample, list):
                raise TypeError(
                    f"La muestra {sample_index} "
                    "debe ser una lista"
                )

            if len(sample) != self.in_features:
                raise ValueError(
                    f"La muestra {sample_index} tiene "
                    f"{len(sample)} características, "
                    f"pero la capa espera "
                    f"{self.in_features}"
                )

    def _validate_weight(self, weight):
        if not isinstance(weight, list):
            raise TypeError(
                "weight debe ser una lista"
            )

        if len(weight) != self.out_features:
            raise ValueError(
                f"weight debe tener "
                f"{self.out_features} filas, "
                f"pero tiene {len(weight)}"
            )

        for row_index in range(len(weight)):
            row = weight[row_index]

            if not isinstance(row, list):
                raise TypeError(
                    f"La fila {row_index} de weight "
                    "debe ser una lista"
                )

            if len(row) != self.in_features:
                raise ValueError(
                    f"La fila {row_index} de weight "
                    f"debe tener {self.in_features} "
                    f"columnas, pero tiene {len(row)}"
                )

    def _validate_bias(self, bias):
        if self.use_bias:
            if bias is None:
                raise ValueError(
                    "La capa utiliza bias, pero no "
                    "se proporcionó ningún bias"
                )

            if not isinstance(bias, list):
                raise TypeError(
                    "bias debe ser una lista"
                )

            if len(bias) != self.out_features:
                raise ValueError(
                    f"bias debe tener "
                    f"{self.out_features} valores, "
                    f"pero tiene {len(bias)}"
                )

        else:
            if bias is not None:
                raise ValueError(
                    "La capa fue creada sin bias, "
                    "pero se proporcionó uno"
                )

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return (
            f"ManualLinear("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
            f")"
        )