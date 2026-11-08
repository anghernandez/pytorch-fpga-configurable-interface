class ManualElementWise:

    def __init__(self, operation):

        valid_operations = [
            "add",
            "sub",
            "mul",
            "div"
        ]

        if not isinstance(operation, str):
            raise TypeError(
                "operation debe ser una cadena de texto."
            )

        operation = operation.lower()

        if operation not in valid_operations:
            raise ValueError(
                "Operación no válida. "
                "Las operaciones permitidas son: "
                "'add', 'sub', 'mul' y 'div'."
            )

        self.operation = operation

    def forward(self, input1, input2):

        return self._elementwise_recursive(
            input1,
            input2
        )

    def _elementwise_recursive(
        self,
        input1,
        input2
    ):

        input1_is_list = isinstance(input1, list)
        input2_is_list = isinstance(input2, list)

        # ====================================================
        # CASO 1:
        # Las dos entradas todavía son listas.
        # ====================================================

        if input1_is_list and input2_is_list:

            if len(input1) != len(input2):
                raise ValueError(
                    "Las entradas deben tener la misma forma."
                )

            output = []

            for i in range(len(input1)):

                value = self._elementwise_recursive(
                    input1[i],
                    input2[i]
                )

                output.append(value)

            return output

        # ====================================================
        # CASO 2:
        # Una entrada es lista y la otra es un número.
        # ====================================================

        if input1_is_list != input2_is_list:
            raise ValueError(
                "Las entradas deben tener la misma estructura."
            )

        # ====================================================
        # CASO 3:
        # Llegamos a los valores individuales.
        # ====================================================

        if isinstance(input1, bool):
            raise TypeError(
                "input1 contiene un booleano."
            )

        if isinstance(input2, bool):
            raise TypeError(
                "input2 contiene un booleano."
            )

        if not isinstance(input1, (int, float)):
            raise TypeError(
                "input1 contiene un valor no numérico."
            )

        if not isinstance(input2, (int, float)):
            raise TypeError(
                "input2 contiene un valor no numérico."
            )

        # ====================================================
        # SELECCIÓN DE LA OPERACIÓN
        # ====================================================

        if self.operation == "add":
            return input1 + input2

        elif self.operation == "sub":
            return input1 - input2

        elif self.operation == "mul":
            return input1 * input2

        elif self.operation == "div":

            if input2 == 0:
                raise ZeroDivisionError(
                    "No se puede dividir entre cero."
                )

            return input1 / input2