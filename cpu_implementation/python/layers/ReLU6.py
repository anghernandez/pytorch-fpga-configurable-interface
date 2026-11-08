# Implementación de ManualReLU6

class ManualReLU6:
    """
    Función de activación ReLU6 manual.

    Calcula elemento por elemento:

        y = 0,   si x < 0
        y = x,   si 0 <= x <= 6
        y = 6,   si x > 6
    """

    def forward(self, x):
        """
        Ejecuta el forward de ReLU6.

        La entrada puede tener cualquier número
        de dimensiones, siempre que esté representada
        mediante listas anidadas de Python.
        """

        self._validate_input(x)

        return self._apply_relu6(x)

    def _apply_relu6(self, x):
        """
        Recorre recursivamente las listas hasta
        encontrar cada valor numérico.
        """

        if isinstance(x, list):

            output = []

            for element_index in range(len(x)):

                element = x[element_index]

                output_element = self._apply_relu6(
                    element
                )

                output.append(
                    output_element
                )

            return output

        if x < 0:
            return 0.0

        if x > 6:
            return 6.0

        return x

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
        return "ManualReLU6()"