# Implementación de ManualReLU

class ManualReLU:
    """
    Función de activación ReLU manual.

    Calcula elemento por elemento:

        y = x,   si x > 0
        y = 0,   si x <= 0
    """

    def forward(self, x):
        """
        Ejecuta el forward de ReLU.

        La entrada puede tener cualquier número
        de dimensiones, siempre que esté representada
        mediante listas anidadas de Python.

        Ejemplos:

            (features)

            (batch_size, features)

            (batch_size, channels, height, width)
        """

        self._validate_input(x)

        return self._apply_relu(x)

    def _apply_relu(self, x):
        """
        Recorre recursivamente las listas hasta
        encontrar cada valor numérico.
        """

        if isinstance(x, list):
            output = []

            for element_index in range(len(x)):
                element = x[element_index]

                output_element = self._apply_relu(
                    element
                )

                output.append(
                    output_element
                )

            return output

        if x > 0:
            return x

        return 0.0

    def _validate_input(self, x):
        """
        Verifica que la entrada sea una lista
        no vacía y que todos sus elementos finales
        sean valores numéricos.
        """

        if not isinstance(x, list):
            raise TypeError(
                "La entrada x debe ser una lista"
            )

        if len(x) == 0:
            raise ValueError(
                "La entrada no puede estar vacía"
            )

        self._validate_elements(x)

    def _validate_elements(self, x):
        """
        Valida recursivamente cada elemento
        de la entrada.
        """

        if isinstance(x, list):
            if len(x) == 0:
                raise ValueError(
                    "La entrada no puede contener "
                    "listas vacías"
                )

            for element_index in range(len(x)):
                self._validate_elements(
                    x[element_index]
                )

            return

        if not isinstance(x, (int, float)):
            raise TypeError(
                "Todos los elementos de la entrada "
                "deben ser valores numéricos"
            )

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "ManualReLU()"