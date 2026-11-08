import math


class ManualBatchNorm2d:

    def __init__(
        self,
        num_features,
        eps=1e-5,
        affine=True,
        bias=True
    ):

        if not isinstance(num_features, int) or num_features <= 0:
            raise ValueError(
                "num_features debe ser un entero mayor que 0"
            )

        if eps <= 0:
            raise ValueError(
                "eps debe ser mayor que 0"
            )

        if not isinstance(affine, bool):
            raise TypeError(
                "affine debe ser True o False"
            )

        if not isinstance(bias, bool):
            raise TypeError(
                "bias debe ser True o False"
            )

        self.num_features = num_features
        self.eps = float(eps)
        self.affine = affine
        self.use_bias = bias if affine else False

        # Parámetros affine
        if self.affine:

            self.weight = [
                1.0
                for _ in range(self.num_features)
            ]

            if self.use_bias:
                self.bias = [
                    0.0
                    for _ in range(self.num_features)
                ]
            else:
                self.bias = None

        else:

            self.weight = None
            self.bias = None

        # Estadísticas usadas durante inferencia
        self.running_mean = [
            0.0
            for _ in range(self.num_features)
        ]

        self.running_var = [
            1.0
            for _ in range(self.num_features)
        ]


    def _get_input_shape(self, x):

        """
        Entrada:

            x[batch][channel][row][column]

        Formato NCHW
        """

        if not isinstance(x, list) or len(x) == 0:
            raise ValueError(
                "La entrada debe ser una lista no vacía"
            )

        batch_size = len(x)

        if (
            not isinstance(x[0], list)
            or len(x[0]) == 0
        ):
            raise ValueError(
                "Cada elemento del batch debe contener canales"
            )

        input_channels = len(x[0])

        if input_channels != self.num_features:
            raise ValueError(
                f"Se esperaban {self.num_features} canales, "
                f"pero se recibieron {input_channels}"
            )

        if (
            not isinstance(x[0][0], list)
            or len(x[0][0]) == 0
        ):
            raise ValueError(
                "Cada canal debe contener filas"
            )

        input_height = len(x[0][0])

        if (
            not isinstance(x[0][0][0], list)
            or len(x[0][0][0]) == 0
        ):
            raise ValueError(
                "Cada fila debe contener columnas"
            )

        input_width = len(x[0][0][0])

        for batch_index in range(batch_size):

            if len(x[batch_index]) != input_channels:
                raise ValueError(
                    "Todas las imágenes deben tener "
                    "el mismo número de canales"
                )

            for channel_index in range(input_channels):

                if (
                    len(
                        x[batch_index][channel_index]
                    )
                    != input_height
                ):
                    raise ValueError(
                        "Todos los canales deben tener "
                        "la misma altura"
                    )

                for row_index in range(input_height):

                    if (
                        len(
                            x[batch_index]
                            [channel_index]
                            [row_index]
                        )
                        != input_width
                    ):
                        raise ValueError(
                            "Todas las filas deben tener "
                            "el mismo ancho"
                        )

        return (
            batch_size,
            input_channels,
            input_height,
            input_width,
        )


    def load_parameters(
        self,
        weight,
        bias,
        running_mean,
        running_var
    ):

        if self.affine:

            if weight is None:
                raise ValueError(
                    "La capa utiliza affine=True, "
                    "pero weight es None"
                )

            if len(weight) != self.num_features:
                raise ValueError(
                    "weight tiene un tamaño incorrecto"
                )

            self.weight = weight.copy()

            if self.use_bias:

                if bias is None:
                    raise ValueError(
                        "La capa utiliza bias=True, "
                        "pero bias es None"
                    )

                if len(bias) != self.num_features:
                    raise ValueError(
                        "bias tiene un tamaño incorrecto"
                    )

                self.bias = bias.copy()

            else:

                if bias is not None:
                    raise ValueError(
                        "La capa fue creada con bias=False"
                    )

                self.bias = None

        else:

            if weight is not None:
                raise ValueError(
                    "La capa fue creada con affine=False"
                )

            if bias is not None:
                raise ValueError(
                    "La capa fue creada con affine=False"
                )

        if running_mean is None:
            raise ValueError(
                "running_mean no puede ser None"
            )

        if running_var is None:
            raise ValueError(
                "running_var no puede ser None"
            )

        if len(running_mean) != self.num_features:
            raise ValueError(
                "running_mean tiene un tamaño incorrecto"
            )

        if len(running_var) != self.num_features:
            raise ValueError(
                "running_var tiene un tamaño incorrecto"
            )

        for channel_index in range(self.num_features):

            if running_var[channel_index] < 0:
                raise ValueError(
                    "running_var no puede contener "
                    "valores negativos"
                )

        self.running_mean = running_mean.copy()
        self.running_var = running_var.copy()


    def forward(self, x):

        (
            batch_size,
            input_channels,
            input_height,
            input_width,
        ) = self._get_input_shape(x)

        output = []

        for batch_index in range(batch_size):

            batch_output = []

            for channel_index in range(input_channels):

                channel_output = []

                mean = self.running_mean[
                    channel_index
                ]

                variance = self.running_var[
                    channel_index
                ]

                denominator = math.sqrt(
                    variance + self.eps
                )

                if self.affine:

                    gamma = self.weight[
                        channel_index
                    ]

                    if self.bias is not None:
                        beta = self.bias[
                            channel_index
                        ]
                    else:
                        beta = 0.0

                else:

                    gamma = 1.0
                    beta = 0.0

                for row_index in range(input_height):

                    output_row = []

                    for column_index in range(input_width):

                        input_value = (
                            x[batch_index]
                            [channel_index]
                            [row_index]
                            [column_index]
                        )

                        normalized_value = (
                            input_value - mean
                        ) / denominator

                        output_value = (
                            gamma * normalized_value
                            + beta
                        )

                        output_row.append(
                            output_value
                        )

                    channel_output.append(
                        output_row
                    )

                batch_output.append(
                    channel_output
                )

            output.append(
                batch_output
            )

        return output


    def __call__(self, x):
        return self.forward(x)


    def __repr__(self):
        return (
            f"ManualBatchNorm2d("
            f"num_features={self.num_features}, "
            f"eps={self.eps}, "
            f"affine={self.affine}, "
            f"bias={self.bias is not None}"
            f")"
        )