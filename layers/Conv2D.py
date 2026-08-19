import math
import random




class ManualConv2d:

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):

        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError("in_channels debe ser un entero mayor que 0")

        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError("out_channels debe ser un entero mayor que 0")

        if not isinstance(groups, int) or groups <= 0:
            raise ValueError("groups debe ser un entero mayor que 0")

        if in_channels % groups != 0:
            raise ValueError(
                "in_channels debe ser divisible entre groups"
            )

        if out_channels % groups != 0:
            raise ValueError(
                "out_channels debe ser divisible entre groups"
            )

        if not isinstance(bias, bool):
            raise TypeError("bias debe ser True o False")

        self.kernel_size = self._to_pair(
            kernel_size,
            "kernel_size",
            allow_zero=False,
        )

        self.stride = self._to_pair(
            stride,
            "stride",
            allow_zero=False,
        )

        self.padding = self._to_pair(
            padding,
            "padding",
            allow_zero=True,
        )

        self.dilation = self._to_pair(
            dilation,
            "dilation",
            allow_zero=False,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.use_bias = bias
        self.padding_mode = padding_mode

        if self.dilation != (1, 1):
            raise NotImplementedError(
                "La primera versión solo admite dilation=1"
            )

        if self.groups != 1:
            raise NotImplementedError(
                "La primera versión solo admite groups=1"
            )

        if self.padding_mode != "zeros":
            raise NotImplementedError(
                "La primera versión solo admite padding_mode='zeros'"
            )

        self.weight = self._initialize_weights()

        if self.use_bias:
            self.bias = self._initialize_bias()
        else:
            self.bias = None

    @staticmethod
    def _to_pair(value, parameter_name, allow_zero):

        if isinstance(value, int):
            pair = (value, value)

        elif (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], int)
            and isinstance(value[1], int)
        ):
            pair = value

        else:
            raise TypeError(
                f"{parameter_name} debe ser un entero "
                "o una tupla de dos enteros"
            )

        minimum = 0 if allow_zero else 1

        if pair[0] < minimum or pair[1] < minimum:
            raise ValueError(
                f"{parameter_name} contiene valores inválidos"
            )

        return pair

    def _calculate_bound(self):

        kernel_height, kernel_width = self.kernel_size

        channels_per_group = (
            self.in_channels // self.groups
        )

        fan_in = (
            channels_per_group
            * kernel_height
            * kernel_width
        )

        return 1.0 / math.sqrt(fan_in)

    def _initialize_weights(self):

        kernel_height, kernel_width = self.kernel_size

        channels_per_group = (
            self.in_channels // self.groups
        )

        bound = self._calculate_bound()

        weights = []

        for output_channel in range(self.out_channels):

            output_filter = []

            for input_channel in range(channels_per_group):

                channel_kernel = []

                for kernel_row in range(kernel_height):

                    row = []

                    for kernel_column in range(kernel_width):

                        value = random.uniform(
                            -bound,
                            bound,
                        )

                        row.append(value)

                    channel_kernel.append(row)

                output_filter.append(channel_kernel)

            weights.append(output_filter)

        return weights

    def _initialize_bias(self):

        bound = self._calculate_bound()

        bias_values = []

        for output_channel in range(self.out_channels):

            value = random.uniform(
                -bound,
                bound,
            )

            bias_values.append(value)

        return bias_values

    def _get_input_shape(self, x):
        """
        La entrada debe tener la estructura:

        x[batch][channel][row][column]
        """

        if not isinstance(x, list) or len(x) == 0:
            raise ValueError(
                "La entrada debe ser una lista no vacía"
            )

        batch_size = len(x)

        if not isinstance(x[0], list) or len(x[0]) == 0:
            raise ValueError(
                "Cada elemento del batch debe contener canales"
            )

        input_channels = len(x[0])

        if input_channels != self.in_channels:
            raise ValueError(
                f"Se esperaban {self.in_channels} canales, "
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

            for input_channel in range(input_channels):

                if (
                    len(x[batch_index][input_channel])
                    != input_height
                ):
                    raise ValueError(
                        "Todos los canales deben tener "
                        "la misma altura"
                    )

                for input_row in range(input_height):

                    if (
                        len(
                            x[batch_index]
                            [input_channel]
                            [input_row]
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

    def _calculate_output_shape(
        self,
        input_height,
        input_width,
    ):

        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        padding_height, padding_width = self.padding
        dilation_height, dilation_width = self.dilation

        output_height = (
            (
                input_height
                + 2 * padding_height
                - dilation_height * (kernel_height - 1)
                - 1
            )
            // stride_height
        ) + 1

        output_width = (
            (
                input_width
                + 2 * padding_width
                - dilation_width * (kernel_width - 1)
                - 1
            )
            // stride_width
        ) + 1

        if output_height <= 0 or output_width <= 0:
            raise ValueError(
                "El kernel es demasiado grande para la entrada "
                "y la configuración seleccionada"
            )

        return output_height, output_width

    def forward(self, x):

        (
            batch_size,
            input_channels,
            input_height,
            input_width,
        ) = self._get_input_shape(x)

        output_height, output_width = (
            self._calculate_output_shape(
                input_height,
                input_width,
            )
        )

        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        padding_height, padding_width = self.padding
        dilation_height, dilation_width = self.dilation

        output = []

        for batch_index in range(batch_size):

            batch_output = []

            for output_channel in range(self.out_channels):

                output_feature_map = []

                for output_row in range(output_height):

                    output_row_values = []

                    for output_column in range(output_width):

                        if self.bias is not None:
                            accumulated_value = (
                                self.bias[output_channel]
                            )
                        else:
                            accumulated_value = 0.0

                        for input_channel in range(input_channels):

                            for kernel_row in range(kernel_height):

                                input_row = (
                                    output_row * stride_height
                                    + kernel_row * dilation_height
                                    - padding_height
                                )

                                for kernel_column in range(
                                    kernel_width
                                ):

                                    input_column = (
                                        output_column * stride_width
                                        + kernel_column
                                        * dilation_width
                                        - padding_width
                                    )

                                    if (
                                        0 <= input_row < input_height
                                        and
                                        0 <= input_column < input_width
                                    ):
                                        input_value = (
                                            x[batch_index]
                                            [input_channel]
                                            [input_row]
                                            [input_column]
                                        )

                                        weight_value = (
                                            self.weight
                                            [output_channel]
                                            [input_channel]
                                            [kernel_row]
                                            [kernel_column]
                                        )

                                        accumulated_value += (
                                            input_value
                                            * weight_value
                                        )

                        output_row_values.append(
                            accumulated_value
                        )

                    output_feature_map.append(
                        output_row_values
                    )

                batch_output.append(
                    output_feature_map
                )

            output.append(batch_output)

        return output

    def __call__(self, x):
        return self.forward(x)