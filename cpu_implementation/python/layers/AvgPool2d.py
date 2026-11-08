
class ManualAvgPool2d:

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):

        self.kernel_size = self._to_pair(
            kernel_size,
            "kernel_size",
            allow_zero=False,
        )

        if stride is None:
            self.stride = self.kernel_size
        else:
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

        if not isinstance(ceil_mode, bool):
            raise TypeError(
                "ceil_mode debe ser True o False"
            )

        if not isinstance(count_include_pad, bool):
            raise TypeError(
                "count_include_pad debe ser True o False"
            )

        if divisor_override is not None:

            if (
                not isinstance(divisor_override, int)
                or isinstance(divisor_override, bool)
            ):
                raise TypeError(
                    "divisor_override debe ser un entero o None"
                )

            if divisor_override == 0:
                raise ValueError(
                    "divisor_override no puede ser 0"
                )

        kernel_height, kernel_width = self.kernel_size
        padding_height, padding_width = self.padding

        if (
            2 * padding_height > kernel_height
            or 2 * padding_width > kernel_width
        ):
            raise ValueError(
                "padding debe ser como máximo la mitad "
                "del tamaño del kernel"
            )

        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    @staticmethod
    def _to_pair(value, parameter_name, allow_zero):

        if isinstance(value, bool):
            raise TypeError(
                f"{parameter_name} debe contener enteros"
            )

        if isinstance(value, int):
            pair = (value, value)

        elif (
            isinstance(value, tuple)
            and len(value) == 1
            and isinstance(value[0], int)
            and not isinstance(value[0], bool)
        ):
            pair = (value[0], value[0])

        elif (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], int)
            and isinstance(value[1], int)
            and not isinstance(value[0], bool)
            and not isinstance(value[1], bool)
        ):
            pair = value

        else:
            raise TypeError(
                f"{parameter_name} debe ser un entero "
                "o una tupla de uno o dos enteros"
            )

        minimum = 0 if allow_zero else 1

        if pair[0] < minimum or pair[1] < minimum:
            raise ValueError(
                f"{parameter_name} contiene valores inválidos"
            )

        return pair



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

            if (
                not isinstance(x[batch_index], list)
                or len(x[batch_index]) != input_channels
            ):
                raise ValueError(
                    "Todas las imágenes deben tener "
                    "el mismo número de canales"
                )

            for input_channel in range(input_channels):

                if (
                    not isinstance(
                        x[batch_index][input_channel],
                        list,
                    )
                    or len(
                        x[batch_index][input_channel]
                    ) != input_height
                ):
                    raise ValueError(
                        "Todos los canales deben tener "
                        "la misma altura"
                    )

                for input_row in range(input_height):

                    if (
                        not isinstance(
                            x[batch_index]
                            [input_channel]
                            [input_row],
                            list,
                        )
                        or len(
                            x[batch_index]
                            [input_channel]
                            [input_row]
                        ) != input_width
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

        height_numerator = (
            input_height
            + 2 * padding_height
            - kernel_height
        )

        width_numerator = (
            input_width
            + 2 * padding_width
            - kernel_width
        )

        if self.ceil_mode:

            output_height = (
                (
                    height_numerator
                    + stride_height
                    - 1
                )
                // stride_height
            ) + 1

            output_width = (
                (
                    width_numerator
                    + stride_width
                    - 1
                )
                // stride_width
            ) + 1

            # PyTorch ignora ventanas que comienzan
            # completamente dentro del padding derecho/inferior.
            if (
                (output_height - 1) * stride_height
                >= input_height + padding_height
            ):
                output_height -= 1

            if (
                (output_width - 1) * stride_width
                >= input_width + padding_width
            ):
                output_width -= 1

        else:

            output_height = (
                height_numerator // stride_height
            ) + 1

            output_width = (
                width_numerator // stride_width
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

        output = []

        for batch_index in range(batch_size):

            batch_output = []

            for input_channel in range(input_channels):

                output_feature_map = []

                for output_row in range(output_height):

                    output_row_values = []

                    for output_column in range(output_width):

                        window_start_row = (
                            output_row * stride_height
                            - padding_height
                        )

                        window_start_column = (
                            output_column * stride_width
                            - padding_width
                        )

                        padded_end_row = min(
                            window_start_row + kernel_height,
                            input_height + padding_height,
                        )

                        padded_end_column = min(
                            window_start_column + kernel_width,
                            input_width + padding_width,
                        )

                        pool_height = (
                            padded_end_row
                            - window_start_row
                        )

                        pool_width = (
                            padded_end_column
                            - window_start_column
                        )

                        pool_size = pool_height * pool_width

                        valid_start_row = max(
                            window_start_row,
                            0,
                        )

                        valid_start_column = max(
                            window_start_column,
                            0,
                        )

                        valid_end_row = min(
                            padded_end_row,
                            input_height,
                        )

                        valid_end_column = min(
                            padded_end_column,
                            input_width,
                        )

                        accumulated_value = 0.0

                        for input_row in range(
                            valid_start_row,
                            valid_end_row,
                        ):

                            for input_column in range(
                                valid_start_column,
                                valid_end_column,
                            ):

                                accumulated_value += (
                                    x[batch_index]
                                    [input_channel]
                                    [input_row]
                                    [input_column]
                                )

                        valid_height = (
                            valid_end_row
                            - valid_start_row
                        )

                        valid_width = (
                            valid_end_column
                            - valid_start_column
                        )

                        valid_value_count = (
                            valid_height * valid_width
                        )

                        if self.divisor_override is not None:

                            divisor = self.divisor_override

                        elif self.count_include_pad:

                            divisor = pool_size

                        else:

                            divisor = valid_value_count

                        average_value = (
                            accumulated_value / divisor
                        )

                        output_row_values.append(
                            average_value
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


if __name__ == "__main__":

    x = [
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]
        ]
    ]

    avg_pool = ManualAvgPool2d(
        kernel_size=2,
        stride=2,
    )

    result = avg_pool(x)

    expected = [
        [
            [
                [3.5, 5.5],
                [11.5, 13.5],
            ]
        ]
    ]

    print("Resultado:", result)
    print("Esperado: ", expected)

    if result == expected:
        print("Prueba: SUPERADA")
    else:
        print("Prueba: FALLIDA")