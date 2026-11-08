class ManualGlobalAvgPool2d:
    """
    Global Average Pooling 2D manual.

    Para cada elemento del batch y para cada canal,
    calcula el promedio de todos los valores espaciales:

        output[n][c][0][0] =
            sum(input[n][c][h][w]) / (H * W)

    Entrada:
        [batch][channel][height][width]

    Salida:
        [batch][channel][1][1]
    """

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

                accumulated_value = 0.0

                for row_index in range(input_height):

                    for column_index in range(input_width):

                        accumulated_value += (
                            x[batch_index]
                            [channel_index]
                            [row_index]
                            [column_index]
                        )

                number_of_values = (
                    input_height * input_width
                )

                average_value = (
                    accumulated_value / number_of_values
                )

                channel_output = [
                    [average_value]
                ]

                batch_output.append(
                    channel_output
                )

            output.append(
                batch_output
            )

        return output

    def _get_input_shape(self, x):
        """
        Verifica que la entrada tenga forma NCHW
        y que todas las dimensiones sean regulares.
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

            for channel_index in range(input_channels):

                if (
                    not isinstance(
                        x[batch_index][channel_index],
                        list,
                    )
                    or len(
                        x[batch_index][channel_index]
                    ) != input_height
                ):
                    raise ValueError(
                        "Todos los canales deben tener "
                        "la misma altura"
                    )

                for row_index in range(input_height):

                    if (
                        not isinstance(
                            x[batch_index]
                            [channel_index]
                            [row_index],
                            list,
                        )
                        or len(
                            x[batch_index]
                            [channel_index]
                            [row_index]
                        ) != input_width
                    ):
                        raise ValueError(
                            "Todas las filas deben tener "
                            "el mismo ancho"
                        )

                    for column_index in range(input_width):

                        value = (
                            x[batch_index]
                            [channel_index]
                            [row_index]
                            [column_index]
                        )

                        if isinstance(value, bool):
                            raise TypeError(
                                "La entrada contiene "
                                "un valor booleano"
                            )

                        if not isinstance(
                            value,
                            (int, float),
                        ):
                            raise TypeError(
                                "Todos los valores de entrada "
                                "deben ser numéricos"
                            )

        return (
            batch_size,
            input_channels,
            input_height,
            input_width,
        )

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "ManualGlobalAvgPool2d()"